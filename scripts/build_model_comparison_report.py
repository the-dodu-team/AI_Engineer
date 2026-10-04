"""Build the RTMDet versus latest-detector comparison report."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
COMBINED_CSV = ROOT / "docs/reports/coco-extended-detectors/comparison.csv"
BY_SOURCE_CSV = ROOT / "docs/reports/coco-extended-detectors/by_source.csv"
OUTPUT = ROOT / "docs/reports/rtmdet-vs-latest"
README_OUTPUT = ROOT / "README.md"

RTMDET_MODELS = ["rtmdet-tiny", "rtmdet-s", "rtmdet-m", "rtmdet-l", "rtmdet-x"]
CANDIDATE_MODELS = [
    "rtdetrv2-r50",
    "rf-detr-large",
    "dfine-l-obj2coco",
    "lw-detr-large",
    "rf-detr-medium",
    "yolov13l",
]
DECISION_MODELS = [*CANDIDATE_MODELS, *RTMDET_MODELS]

DISPLAY = {
    "rtdetrv2-r50": "RT-DETRv2-R50",
    "dfine-l-obj2coco": "D-FINE-L",
    "rf-detr-large": "RF-DETR Large",
    "rf-detr-medium": "RF-DETR Medium",
    "lw-detr-large": "LW-DETR Large",
    "yolov13l": "YOLOv13-L",
    "rtmdet-tiny": "RTMDet-tiny",
    "rtmdet-s": "RTMDet-s",
    "rtmdet-m": "RTMDet-m",
    "rtmdet-l": "RTMDet-l",
    "rtmdet-x": "RTMDet-x",
}

BLUE = "#2563EB"
NAVY = "#0F172A"
SLATE = "#94A3B8"
RED = "#DC2626"
GREEN = "#059669"
GRID = "#E2E8F0"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pct(value: str) -> float:
    return float(value) * 100


def setup_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Malgun Gothic", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": NAVY,
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "text.color": NAVY,
        }
    )


def color_for(model: str) -> str:
    if model in RTMDET_MODELS:
        return SLATE
    return BLUE


def save_overview(rows: dict[str, dict[str, str]]) -> None:
    ordered = sorted(DECISION_MODELS, key=lambda model: float(rows[model]["accuracy"]))
    names = [DISPLAY[m] for m in ordered]
    accuracies = [pct(rows[m]["accuracy"]) for m in ordered]
    fnrs = [pct(rows[m]["fnr"]) for m in ordered]
    fprs = [pct(rows[m]["fpr"]) for m in ordered]
    fig, (ax_accuracy, ax_error) = plt.subplots(1, 2, figsize=(18, 8.5), layout="constrained")

    fig.suptitle("RTMDet 모델 크기별 결과 vs 탐지 모델 후보", fontsize=23, fontweight="bold")
    bars = ax_accuracy.barh(names, accuracies, color=[color_for(m) for m in ordered], height=0.68)
    ax_accuracy.set_xlim(85, 97)
    ax_accuracy.set_xlabel("정확도 (%)")
    ax_accuracy.set_title("A. 전체 정확도", loc="left", fontsize=15, fontweight="bold")
    ax_accuracy.grid(axis="x", color=GRID, linewidth=0.8)
    ax_accuracy.set_axisbelow(True)
    for bar, value in zip(bars, accuracies):
        ax_accuracy.text(value + 0.12, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center", fontweight="bold", fontsize=10)
    for spine in ax_accuracy.spines.values():
        spine.set_visible(False)

    y = np.arange(len(names))
    width = 0.38
    fnr_bars = ax_error.barh(y - width / 2, fnrs, height=width, color=RED, label="FNR: 대상 누락")
    fpr_bars = ax_error.barh(y + width / 2, fprs, height=width, color=GREEN, label="FPR: 비대상 오수락")
    ax_error.axvline(5, color=NAVY, linestyle="--", linewidth=1.5, label="목표 5%")
    ax_error.set_yticks(y, names)
    ax_error.set_xlim(0, 17)
    ax_error.set_xlabel("오류율 (%) — 낮을수록 좋음")
    ax_error.set_title("B. FNR과 FPR", loc="left", fontsize=15, fontweight="bold")
    ax_error.grid(axis="x", color=GRID, linewidth=0.8)
    ax_error.set_axisbelow(True)
    ax_error.legend(loc="lower right", frameon=False, fontsize=10)
    for bars_to_label in (fnr_bars, fpr_bars):
        for bar in bars_to_label:
            value = bar.get_width()
            ax_error.text(value + 0.15, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center", fontsize=8)
    for spine in ax_error.spines.values():
        spine.set_visible(False)

    fig.savefig(OUTPUT / "overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_latency(rows: dict[str, dict[str, str]]) -> None:
    ordered = sorted(DECISION_MODELS, key=lambda model: float(rows[model]["latency_ms"]))
    names = [DISPLAY[model] for model in ordered]
    latencies = [float(rows[model]["latency_ms"]) for model in ordered]
    colors = [BLUE for _ in ordered]

    fig, axis = plt.subplots(figsize=(12, 7.5), layout="constrained")
    bars = axis.barh(names, latencies, color=colors, height=0.68)
    axis.invert_yaxis()
    axis.set_xlim(0, 1500)
    axis.set_xlabel("CPU 레이턴시 (ms/image) — 낮을수록 좋음", fontweight="bold")
    axis.set_title("전체 모델 CPU 레이턴시", loc="left", fontsize=18, fontweight="bold")
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for bar, latency in zip(bars, latencies):
        axis.text(latency + 15, bar.get_y() + bar.get_height() / 2, f"{latency:.1f} ms", va="center", fontweight="bold")
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.savefig(OUTPUT / "latency.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


CLASSES = ["computer", "book", "other"]


def build_confusion_matrices(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    matrices = {model: np.zeros((3, 3), dtype=int) for model in DECISION_MODELS}
    class_index = {label: index for index, label in enumerate(CLASSES)}
    for row in rows:
        model = row["model"]
        if model not in matrices:
            continue
        source = row["source"]
        true_label = "computer" if source in {"computer", "laptop"} else "book" if source == "book" else "other"
        true_index = class_index[true_label]
        for predicted_label in CLASSES:
            matrices[model][true_index, class_index[predicted_label]] += int(row[predicted_label])
    for model, matrix in matrices.items():
        if int(matrix.sum()) != 372:
            raise RuntimeError(f"{model} confusion matrix has {matrix.sum()} rows, expected 372")
    return matrices


def class_metrics(matrix: np.ndarray) -> list[dict[str, float | int | str]]:
    total = int(matrix.sum())
    result = []
    for index, label in enumerate(CLASSES):
        tp = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted = int(matrix[:, index].sum())
        fp = predicted - tp
        negatives = total - support
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        fpr = fp / negatives if negatives else 0.0
        result.append(
            {
                "class": label,
                "support": support,
                "tp": tp,
                "misses": support - tp,
                "recall": recall,
                "precision": precision,
                "fpr": fpr,
                "fnr": 1 - recall,
            }
        )
    return result


def save_confusion_matrices(matrices: dict[str, np.ndarray], rows: dict[str, dict[str, str]]) -> None:
    ordered = sorted(DECISION_MODELS, key=lambda model: float(rows[model]["accuracy"]), reverse=True)
    fig, axes = plt.subplots(3, 4, figsize=(17, 13))
    image = None
    for axis, model in zip(axes.flat, ordered):
        matrix = matrices[model]
        normalized = matrix / matrix.sum(axis=1, keepdims=True) * 100
        image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=100)
        axis.set_title(DISPLAY[model], fontsize=12, fontweight="bold")
        axis.set_xticks(range(3), CLASSES, rotation=25, ha="right")
        axis.set_yticks(range(3), CLASSES)
        axis.set_xlabel("예측")
        axis.set_ylabel("정답")
        for true_index in range(3):
            for predicted_index in range(3):
                value = normalized[true_index, predicted_index]
                color = "white" if value >= 55 else NAVY
                axis.text(
                    predicted_index,
                    true_index,
                    f"{matrix[true_index, predicted_index]}\n{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                    fontweight="bold" if true_index == predicted_index else "normal",
                )
    for axis in axes.flat[len(ordered) :]:
        axis.axis("off")
    fig.suptitle("모델별 Confusion Matrix", fontsize=22, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.05, top=0.93, wspace=0.34, hspace=0.42)
    fig.savefig(OUTPUT / "confusion-matrices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_class_metrics(matrices: dict[str, np.ndarray], rows: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    ordered = sorted(DECISION_MODELS, key=lambda model: float(rows[model]["accuracy"]), reverse=True)
    metrics_by_model = {model: class_metrics(matrices[model]) for model in ordered}
    recalls = np.array([[float(item["recall"]) * 100 for item in metrics_by_model[model]] for model in ordered])
    fnrs = np.array([[float(item["fnr"]) * 100 for item in metrics_by_model[model]] for model in ordered])
    fprs = np.array([[float(item["fpr"]) * 100 for item in metrics_by_model[model]] for model in ordered])

    fig, axes = plt.subplots(1, 3, figsize=(20, 9), layout="constrained")
    panels = [
        (axes[0], recalls, "클래스별 Recall", "높을수록 좋음", "YlGnBu", 65, 100),
        (axes[1], fnrs, "클래스별 FNR", "낮을수록 좋음", "YlOrRd", 0, max(35, float(fnrs.max()))),
        (axes[2], fprs, "클래스별 FPR", "낮을수록 좋음", "YlOrRd", 0, max(16, float(fprs.max()))),
    ]
    for axis, values, title, subtitle, cmap, vmin, vmax in panels:
        image = axis.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_xticks(range(3), CLASSES)
        axis.set_yticks(range(len(ordered)), [DISPLAY[model] for model in ordered])
        axis.set_title(f"{title}\n{subtitle}", fontsize=15, fontweight="bold")
        axis.add_patch(Rectangle((0.5, -0.5), 1, len(ordered), fill=False, edgecolor=NAVY, linewidth=2.5))
        text_threshold = (vmin + vmax) / 2
        for row_index in range(len(ordered)):
            for class_index in range(3):
                value = values[row_index, class_index]
                text_color = "white" if value >= text_threshold else NAVY
                axis.text(class_index, row_index, f"{value:.1f}%", ha="center", va="center", fontsize=9, fontweight="bold", color=text_color)
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("비율 (%)")
    fig.savefig(OUTPUT / "class-metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    flat_rows: list[dict[str, object]] = []
    for model in ordered:
        for item in metrics_by_model[model]:
            flat_rows.append({"model": model, **item})
    with (OUTPUT / "class-metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    return flat_rows


def build_readme(rows: dict[str, dict[str, str]], class_rows: list[dict[str, object]]) -> None:
    selected = sorted(DECISION_MODELS, key=lambda model: float(rows[model]["accuracy"]), reverse=True)
    class_lookup = {(str(row["model"]), str(row["class"])): row for row in class_rows}
    book_order = sorted(DECISION_MODELS, key=lambda model: float(class_lookup[model, "book"]["recall"]), reverse=True)
    license_labels = {
        "permissive_with_notices": "Apache-2.0 계열",
        "objects365_terms_review": "Objects365 권리 확인 필요",
        "copyleft_or_separate_permission": "AGPL-3.0/별도 허가",
    }

    lines = [
        "# RTMDet 모델 크기별 결과 vs 탐지 모델 후보",
        "",
        "동일한 고유 이미지 372장에서 RTMDet 5종과 탐지 모델 후보 6종을 Top-1 기준으로 비교했습니다.",
        "",
        "- 후보 6종 중 최고값: **RT-DETRv2-R50 95.70%(356/372)**",
        "- RTMDet 5종 중 최고값: **RTMDet-x 94.09%(350/372)**",
        "- 모든 모델의 전체 FPR은 2% 이하였지만, 전체 FNR은 5% 목표를 충족하지 못했습니다.",
        "- 11개 모델 모두 `book` recall이 `computer` recall보다 낮았습니다.",
        "",
        "![전체 정확도와 FNR·FPR](docs/reports/rtmdet-vs-latest/overview.png)",
        "",
        "## 전체 결과",
        "",
        "전체 FNR은 `computer`·`book`을 `other`로 거절한 비율(분모 271), 전체 FPR은 `other`를 대상 클래스로 수락한 비율(분모 101)입니다.",
        "",
        "| 순위 | 모델 | 구분 | 정답/372 | 정확도 | FNR | FPR | CPU 레이턴시¹ | 상업 이용 조건 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for rank, model in enumerate(selected, 1):
        row = rows[model]
        group = "RTMDet" if model in RTMDET_MODELS else "후보"
        commercial = license_labels.get(row["commercial_status"], "확인 필요")
        lines.append(
            f"| {rank} | **{DISPLAY[model]}** | {group} | {row['correct']} | {pct(row['accuracy']):.2f}% | {pct(row['fnr']):.2f}% | {pct(row['fpr']):.2f}% | {float(row['latency_ms']):.1f} ms | {commercial} |"
        )

    lines += [
        "",
        "¹ 이미지 디스크 읽기와 모델 로딩을 제외한 CPU 이미지당 평균입니다. RTMDet과 후보 모델은 실행 프레임워크와 반복 횟수가 달라 모델군 사이의 속도 배수로 해석하지 않습니다.",
        "",
        "## CPU 레이턴시",
        "",
        "전체 11개 모델을 CPU 이미지당 평균 레이턴시가 낮은 순서로 정렬했습니다. 모델군 사이의 런타임 차이가 있으므로 참고값으로 봐야 합니다.",
        "",
        "![전체 모델 CPU 레이턴시](docs/reports/rtmdet-vs-latest/latency.png)",
        "",
        "## 클래스별 결과",
        "",
        "클래스별 FNR은 해당 클래스를 다른 클래스로 놓친 비율(`1 - Recall`)입니다. FPR은 각 클래스를 one-vs-rest로 계산했습니다. 예를 들어 `book FPR`은 실제 `computer` 또는 `other` 이미지를 `book`으로 예측한 비율입니다. `other FPR`은 대상 이미지를 `other`로 예측한 비율이므로 위 전체 FNR과 같습니다.",
        "",
        "![클래스별 Recall, FNR, FPR](docs/reports/rtmdet-vs-latest/class-metrics.png)",
        "",
        "### Book 상세",
        "",
        "`book`은 71장입니다. 최고 recall은 LW-DETR Large의 90.14%(64/71)였으며, 전체 정확도 1위 RT-DETRv2-R50은 85.92%(61/71)였습니다.",
        "",
        "| 모델 | 정답/71 | 누락 | Recall | FNR | Precision | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in book_order:
        item = class_lookup[model, "book"]
        lines.append(
            f"| {DISPLAY[model]} | {item['tp']} | {item['misses']} | {float(item['recall']):.2%} | {float(item['fnr']):.2%} | {float(item['precision']):.2%} | {float(item['fpr']):.2%} |"
        )

    lines += [
        "",
        "### 전체 클래스 Recall·FNR·FPR",
        "",
        "| 모델 | Computer Recall | Computer FNR | Computer FPR | Book Recall | Book FNR | Book FPR | Other Recall | Other FNR | Other FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in selected:
        computer = class_lookup[model, "computer"]
        book = class_lookup[model, "book"]
        other = class_lookup[model, "other"]
        lines.append(
            f"| {DISPLAY[model]} | {float(computer['recall']):.2%} | {float(computer['fnr']):.2%} | {float(computer['fpr']):.2%} | {float(book['recall']):.2%} | {float(book['fnr']):.2%} | {float(book['fpr']):.2%} | {float(other['recall']):.2%} | {float(other['fnr']):.2%} | {float(other['fpr']):.2%} |"
        )

    lines += [
        "",
        "## Confusion Matrix",
        "",
        "각 행은 실제 클래스, 각 열은 예측 클래스입니다. 셀에는 이미지 수와 실제 클래스 내 비율을 함께 표시했습니다.",
        "",
        "![모델별 confusion matrix](docs/reports/rtmdet-vs-latest/confusion-matrices.png)",
        "",
        "## 해석 시 주의사항",
        "",
        "- 결과는 고유 이미지 372장(`computer` 200, `book` 71, `other` 101)의 이미지 단위 Top-1 평가입니다.",
        "- 모델 선정에 사용한 데이터와 같은 평가 세트이므로 새로운 이미지에 대한 일반화 성능은 별도 검증이 필요합니다.",
        "- ¹ RTMDet 5종의 지연은 같은 기존 실행 안에서 비교할 수 있습니다. 후보 모델은 프레임워크와 반복 횟수가 달라 RTMDet과의 속도 배수를 계산하지 않았습니다.",
        "- D-FINE-L과 LW-DETR은 Objects365 관련 가중치 권리를 확인해야 합니다. YOLOv13-L은 AGPL-3.0 준수 또는 별도 허가가 필요합니다.",
        "",
        "## 근거 자료",
        "",
        "- [전체 클래스 수치 CSV](docs/reports/rtmdet-vs-latest/class-metrics.csv)",
        "- [20개 모델 전체 집계](docs/reports/coco-extended-detectors/README.md)",
        "- [통합 수치 CSV](docs/reports/coco-extended-detectors/comparison.csv)",
        "- [평가 및 재현 방법](docs/coco-extended-detectors.md)",
        "",
    ]
    README_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    combined = {row["model"]: row for row in read_rows(COMBINED_CSV)}
    missing = set(DECISION_MODELS) - set(combined)
    if missing:
        raise RuntimeError(f"Missing models in {COMBINED_CSV}: {sorted(missing)}")
    by_source = read_rows(BY_SOURCE_CSV)
    matrices = build_confusion_matrices(by_source)
    setup_plotting()
    save_overview(combined)
    save_latency(combined)
    save_confusion_matrices(matrices, combined)
    class_rows = save_class_metrics(matrices, combined)
    build_readme(combined, class_rows)
    print(README_OUTPUT)


if __name__ == "__main__":
    main()
