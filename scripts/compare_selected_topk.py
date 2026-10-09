"""Compare Top-1 and Top-2 for RTMDet-x and three selected detectors.

The input is produced by evaluate_extended_candidates.py and contains the
maximum score for each task group.  Top-1 and Top-2 are therefore evaluated
from the exact same inference output; no model is run a second time.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("computer", "book", "other")
TARGET_GROUPS = ("computer", "book")
SELECTED_MODELS = ("rtdetrv2-r50", "rtdetrv2-r34", "lw-detr-large")
ALL_MODELS = ("rtmdet-x", *SELECTED_MODELS)
DISPLAY_NAMES = {
    "rtdetrv2-r50": "RT-DETRv2-R50",
    "rtdetrv2-r34": "RT-DETRv2-R34",
    "lw-detr-large": "LW-DETR Large",
    "rtmdet-x": "RTMDet-x",
}
SHORT_NAMES = {
    "rtdetrv2-r50": "RT-DETRv2-R50",
    "rtdetrv2-r34": "RT-DETRv2-R34",
    "lw-detr-large": "LW-DETR Large",
    "rtmdet-x": "RTMDet-x",
}
BLUE = "#2563EB"
CYAN = "#06B6D4"
RED = "#DC2626"
GREEN = "#059669"
SLATE = "#64748B"
NAVY = "#0F172A"
GRID = "#CBD5E1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def accepted_groups(scores: dict[str, float], threshold: float, top_k: int) -> tuple[str, ...]:
    """Return up to k threshold-passing groups ordered by descending score.

    This matches the existing Top-1 rule.  When no group passes the threshold,
    the image is rejected as ``other``.
    """
    if not 1 <= top_k <= len(GROUPS):
        raise ValueError(f"top_k must be between 1 and {len(GROUPS)}")
    passing = [group for group in GROUPS if scores[group] >= threshold]
    ranked = sorted(passing, key=lambda group: (-scores[group], GROUPS.index(group)))
    return tuple(ranked[:top_k]) if ranked else ("other",)


def summarize(rows: Iterable[dict[str, str]], threshold: float, top_k: int) -> tuple[dict, list[dict]]:
    source_rows = list(rows)
    if not source_rows:
        raise ValueError("Prediction file is empty")

    detail: list[dict] = []
    for row in source_rows:
        scores = {group: float(row[f"score_{group}"]) for group in GROUPS}
        accepted = accepted_groups(scores, threshold, top_k)
        true_label = row["true_label"]
        if true_label not in GROUPS:
            raise ValueError(f"Unexpected true label: {true_label}")
        detail.append(
            {
                "path": row["path"],
                "sha256": row.get("sha256", ""),
                "true_label": true_label,
                "top_k": top_k,
                "accepted_labels": ";".join(accepted),
                "contains_truth": true_label in accepted,
                "accepts_any_target": any(group in accepted for group in TARGET_GROUPS),
                **{f"score_{group}": scores[group] for group in GROUPS},
            }
        )

    positives = [row for row in detail if row["true_label"] in TARGET_GROUPS]
    negatives = [row for row in detail if row["true_label"] == "other"]
    false_rejects = sum(not row["accepts_any_target"] for row in positives)
    false_accepts = sum(row["accepts_any_target"] for row in negatives)
    expected_label_misses = sum(not row["contains_truth"] for row in positives)

    class_metrics = {}
    for group in TARGET_GROUPS:
        actual_positive = [row for row in detail if row["true_label"] == group]
        actual_negative = [row for row in detail if row["true_label"] != group]
        tp = sum(group in row["accepted_labels"].split(";") for row in actual_positive)
        fp = sum(group in row["accepted_labels"].split(";") for row in actual_negative)
        class_metrics[group] = {
            "support": len(actual_positive),
            "tp": tp,
            "fn": len(actual_positive) - tp,
            "fp": fp,
            "fnr": 1 - tp / len(actual_positive) if actual_positive else 0.0,
            "fpr": fp / len(actual_negative) if actual_negative else 0.0,
        }

    summary = {
        "top_k": top_k,
        "images": len(detail),
        "truth_inclusion_rate": sum(row["contains_truth"] for row in detail) / len(detail),
        "truth_included": sum(row["contains_truth"] for row in detail),
        "multiple_candidate_rate": sum(";" in row["accepted_labels"] for row in detail) / len(detail),
        "false_accepts": false_accepts,
        "negatives": len(negatives),
        "fpr": false_accepts / len(negatives) if negatives else 0.0,
        "false_rejects": false_rejects,
        "positives": len(positives),
        "fnr": false_rejects / len(positives) if positives else 0.0,
        "expected_label_misses": expected_label_misses,
        "expected_label_fnr": expected_label_misses / len(positives) if positives else 0.0,
        "class_metrics": class_metrics,
    }
    return summary, detail


def validate_predictions(model: str, rows: list[dict[str, str]], manifest: dict) -> None:
    required = {"path", "true_label", *(f"score_{group}" for group in GROUPS)}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"{model} predictions are missing columns: {sorted(missing)}")
    expected = [(row["path"], row["true_label"]) for row in manifest["images"]]
    actual = [(row["path"], row["true_label"]) for row in rows]
    if actual != expected:
        raise ValueError(f"{model} predictions do not match run.json order and labels")


def pct(value: float | str) -> str:
    return "-" if value == "" else f"{float(value):.2%}"


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


def label_bars(axis, bars, suffix="%") -> None:
    for bar in bars:
        value = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(axis.get_ylim()[1] * 0.012, 0.15),
            f"{value:.2f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def save_topk_overview(output: Path, summaries: dict[str, dict]) -> None:
    names = [SHORT_NAMES[model] for model in ALL_MODELS]
    x = np.arange(len(names))
    width = 0.34
    panels = [
        ("truth_inclusion_rate", "Truth inclusion", (90, 101), GREEN),
        ("fnr", "Target FNR", (0, 10), RED),
        ("fpr", "Target FPR", (0, 72), RED),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5), layout="constrained")
    for axis, (metric, title, limits, top2_color) in zip(axes, panels):
        top1 = [summaries[model]["top1"][metric] * 100 for model in ALL_MODELS]
        top2 = [summaries[model]["top2"][metric] * 100 for model in ALL_MODELS]
        bars1 = axis.bar(x - width / 2, top1, width, label="Top-1", color=BLUE)
        bars2 = axis.bar(x + width / 2, top2, width, label="Top-2", color=top2_color)
        axis.set_title(title, loc="left", fontsize=15, fontweight="bold")
        axis.set_xticks(x, names, rotation=14, ha="right")
        axis.set_ylim(*limits)
        axis.set_ylabel("Percent (%)")
        axis.grid(axis="y", color=GRID, linewidth=.8)
        axis.set_axisbelow(True)
        label_bars(axis, bars1)
        label_bars(axis, bars2)
        for spine in axis.spines.values():
            spine.set_visible(False)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Four Detectors: Top-1 vs Top-2", fontsize=22, fontweight="bold")
    fig.savefig(output / "topk-overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_recovery_cost(output: Path, summaries: dict[str, dict]) -> None:
    names = [SHORT_NAMES[model] for model in ALL_MODELS]
    recovered = []
    added_false_accepts = []
    for model in ALL_MODELS:
        top1, top2 = summaries[model]["top1"], summaries[model]["top2"]
        recovered.append(top2["truth_included"] - top1["truth_included"])
        added_false_accepts.append(top2["false_accepts"] - top1["false_accepts"])
    x = np.arange(len(names))
    width = .36
    fig, axis = plt.subplots(figsize=(11, 6.5), layout="constrained")
    good = axis.bar(x - width / 2, recovered, width, color=GREEN, label="Additional truths included")
    cost = axis.bar(x + width / 2, added_false_accepts, width, color=RED, label="Additional false accepts")
    axis.set_xticks(x, names)
    axis.set_ylabel("Images")
    axis.set_title("Top-2 Recovery vs False-Accept Cost", loc="left", fontsize=18, fontweight="bold")
    axis.grid(axis="y", color=GRID, linewidth=.8)
    axis.set_axisbelow(True)
    label_bars(axis, good, suffix="")
    label_bars(axis, cost, suffix="")
    axis.legend(frameon=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.savefig(output / "recovery-vs-cost.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_tradeoff(output: Path, comparison_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), layout="constrained")
    for axis, zoom in zip(axes, (False, True)):
        for row in comparison_rows:
            x, y = float(row["fpr"]) * 100, float(row["fnr"]) * 100
            top2 = int(row["top_k"]) == 2
            color = RED if top2 else BLUE
            marker = "s" if row["model"] == "rtmdet-x" else "^" if top2 else "o"
            axis.scatter(x, y, s=85, color=color, marker=marker, edgecolor="white", linewidth=.8, zorder=3)
            label = SHORT_NAMES.get(row["model"], row["model"])
            label += " k2" if top2 else " k1"
            offset = (5, -14) if row["model"] == "rtmdet-x" and not top2 else (5, 5)
            axis.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=8)
        axis.axvline(5, color=GREEN, linestyle="--", linewidth=1.2)
        axis.axhline(5, color=GREEN, linestyle="--", linewidth=1.2)
        axis.fill_between([0, 5], 0, 5, color=GREEN, alpha=.08)
        axis.set_xlabel("FPR (%) — lower is better")
        axis.set_ylabel("FNR (%) — lower is better")
        axis.grid(color=GRID, linewidth=.7)
        axis.set_axisbelow(True)
        axis.set_title("Full range" if not zoom else "Top-1 zoom", loc="left", fontweight="bold")
        if zoom:
            axis.set_xlim(-.2, 5.2)
            axis.set_ylim(-.5, 17)
        else:
            axis.set_xlim(-2, 72)
            axis.set_ylim(-.5, 17)
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.suptitle("FNR–FPR Operating Trade-off", fontsize=21, fontweight="bold")
    fig.savefig(output / "fnr-fpr-tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_class_heatmaps(output: Path, summaries: dict[str, dict]) -> None:
    row_labels = []
    metric_rows = {metric: [] for metric in ("fnr", "fpr")}
    for model in ALL_MODELS:
        for key in ("top1", "top2"):
            summary = summaries[model][key]
            row_labels.append(f"{SHORT_NAMES[model]} k{summary['top_k']}")
            for metric in metric_rows:
                metric_rows[metric].append(
                    [summary["class_metrics"][group][metric] * 100 for group in TARGET_GROUPS]
                )
    panels = [
        ("fnr", "FNR", "YlOrRd", 0, 35),
        ("fpr", "FPR", "YlOrRd", 0, 25),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 9), layout="constrained")
    for axis, (metric, title, cmap, vmin, vmax) in zip(axes, panels):
        values = np.asarray(metric_rows[metric])
        image = axis.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_xticks(range(2), ["computer", "book"])
        axis.set_yticks(range(len(row_labels)), row_labels)
        axis.set_title(title, loc="left", fontsize=15, fontweight="bold")
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                axis.text(column_index, row_index, f"{value:.1f}%", ha="center", va="center", fontweight="bold")
        fig.colorbar(image, ax=axis, fraction=.046, pad=.04).set_label("Percent (%)")
    fig.suptitle("Class-Level FNR and FPR", fontsize=21, fontweight="bold")
    fig.savefig(output / "class-metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_top1_reference(output: Path, comparison_rows: list[dict]) -> None:
    rows = [row for row in comparison_rows if int(row["top_k"]) == 1]
    rows.sort(key=lambda row: float(row["truth_inclusion_rate"]))
    names = [SHORT_NAMES.get(row["model"], row["model"]) for row in rows]
    colors = [SLATE if row["model"] == "rtmdet-x" else BLUE for row in rows]
    accuracy = [float(row["truth_inclusion_rate"]) * 100 for row in rows]
    latency = [float(row["latency_ms"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), layout="constrained")
    bars = axes[0].barh(names, accuracy, color=colors)
    axes[0].set_xlim(85, 97)
    axes[0].set_xlabel("Top-1 accuracy (%)")
    axes[0].set_title("Accuracy", loc="left", fontweight="bold")
    for bar, value in zip(bars, accuracy):
        axes[0].text(value + .08, bar.get_y() + bar.get_height()/2, f"{value:.2f}%", va="center", fontsize=9)
    bars = axes[1].barh(names, latency, color=colors)
    axes[1].set_xlabel("CPU latency (ms/image)")
    axes[1].set_title("Latency (reference only)", loc="left", fontweight="bold")
    for bar, value in zip(bars, latency):
        axes[1].text(value + 12, bar.get_y() + bar.get_height()/2, f"{value:.0f}", va="center", fontsize=9)
    for axis in axes:
        axis.grid(axis="x", color=GRID, linewidth=.8)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.suptitle("Selected Top-1 vs RTMDet-x", fontsize=21, fontweight="bold")
    fig.savefig(output / "top1-vs-rtmdet.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(output: Path, comparison_rows: list[dict], summaries: dict[str, dict]) -> None:
    metric_note = (
        "<sub>FNR은 실제 대상(computer·book)을 대상 후보 없이 거절한 비율, FPR은 실제 other를 computer 또는 book 후보로 수락한 비율입니다. "
        "CPU 지연은 이미지 읽기·모델 로딩을 제외한 전처리·추론·후처리 평균이며, 런타임과 반복 횟수가 달라 참고값으로만 비교합니다.</sub>"
    )
    class_metric_note = (
        "<sub>클래스별 FNR은 해당 실제 클래스가 후보에 포함되지 않은 비율이며, 클래스별 FPR은 해당 클래스가 아닌 이미지에 "
        "그 클래스를 후보로 잘못 포함한 비율입니다. CPU 지연은 이미지 읽기·모델 로딩을 제외한 전처리·추론·후처리 평균이며, "
        "런타임과 반복 횟수가 달라 참고값으로만 비교합니다.</sub>"
    )
    lines = [
        "# RTMDet-x와 선택 모델 Top-1·Top-2 비교",
        "",
        "RTMDet-x, RT-DETRv2-R50, RT-DETRv2-R34, LW-DETR Large를 동일한 고유 이미지 372장에서 비교합니다.",
        "네 모델 모두 같은 그룹 임계값으로 Top-1과 Top-2를 계산했으며 추가 학습이나 fine-tuning은 수행하지 않았습니다.",
        "",
        "## 결론",
        "",
        "- **자동 단일 판정은 RT-DETRv2-R50 Top-1이 가장 안정적**입니다: 정답 포함률 95.70%, FNR 5.54%, FPR 0.99%.",
        "- 네 모델 모두 Top-2에서 전체 FNR이 0%가 됐지만 FPR은 33.66~65.35%로 증가했습니다.",
        "- Top-2 중 FPR은 LW-DETR Large 33.66%, RTMDet-x 41.58%, RT-DETRv2-R50 53.47%, RT-DETRv2-R34 65.35% 순입니다.",
        "- 따라서 Top-2는 자동 승인보다 후속 분류기 또는 사용자 확인용 shortlist에 적합합니다.",
        "",
        "![네 모델 Top-1과 Top-2 핵심 지표](topk-overview.png)",
        "",
        "## 전체 결과",
        "",
        "정답 포함률은 실제 폴더 정답이 반환 후보 안에 존재하는 비율이며 Top-2 단일 분류 정확도가 아닙니다.",
        "",
        "| 모델 | k | 정답 포함 | 정답 포함률 | 전체 FNR | 전체 FPR | 복수 후보율 | CPU 지연 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    row_lookup = {(row["model"], int(row["top_k"])): row for row in comparison_rows}
    for model in ALL_MODELS:
        for top_k in (1, 2):
            row = row_lookup[model, top_k]
            summary = summaries[model][f"top{top_k}"]
            lines.append(
                f"| {row['display_name']} | {top_k} | {summary['truth_included']}/{summary['images']} | "
                f"{pct(row['truth_inclusion_rate'])} | {pct(row['fnr'])} | {pct(row['fpr'])} | "
                f"{pct(row['multiple_candidate_rate'])} | {float(row['latency_ms']):.1f} ms |"
            )
    lines += ["", metric_note, "", "## Top-2 변화", "", "| 모델 | 추가 정답 포함 | 추가 오수락 | FNR 변화 | FPR 변화 |", "|---|---:|---:|---:|---:|"]
    for model in ALL_MODELS:
        top1, top2 = summaries[model]["top1"], summaries[model]["top2"]
        lines.append(
            f"| {DISPLAY_NAMES[model]} | +{top2['truth_included'] - top1['truth_included']} | "
            f"+{top2['false_accepts'] - top1['false_accepts']} | {top1['fnr']:.2%} → {top2['fnr']:.2%} | "
            f"{top1['fpr']:.2%} → {top2['fpr']:.2%} |"
        )
    lines += [
        "", metric_note, "", "![Top-2 복구량과 오수락 비용](recovery-vs-cost.png)", "",
        "## 클래스별 FNR·FPR", "",
        "| 모델 | k | Computer FNR | Computer FPR | Book FNR | Book FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in ALL_MODELS:
        for key in ("top1", "top2"):
            summary = summaries[model][key]
            computer = summary["class_metrics"]["computer"]
            book = summary["class_metrics"]["book"]
            lines.append(
                f"| {DISPLAY_NAMES[model]} | {summary['top_k']} | {computer['fnr']:.2%} | "
                f"{computer['fpr']:.2%} | {book['fnr']:.2%} | {book['fpr']:.2%} |"
            )
    lines += [
        "", class_metric_note, "", "![클래스별 FNR·FPR](class-metrics.png)", "",
        "## FNR–FPR 트레이드오프", "",
        "왼쪽은 Top-2까지 포함한 전체 범위, 오른쪽은 Top-1 구간을 확대한 그림입니다. 초록색 영역은 FNR·FPR이 모두 5% 이하인 목표 구간입니다.",
        "", "![FNR과 FPR 트레이드오프](fnr-fpr-tradeoff.png)", "",
        "## Top-1 정확도·CPU 지연", "", "![네 모델 Top-1 정확도와 CPU 지연](top1-vs-rtmdet.png)", "",
        metric_note, "", "## 평가 조건", "",
        "- 고유 이미지 372장: computer 200, book 71, other 101",
        "- 그룹 임계값 0.05, 원시 탐지 하한 0.01, 입력 640 또는 체크포인트 기본 전처리",
        "- CPU 10 threads, warm-up 5회",
        "- 선택 모델은 이미지당 1회, RTMDet-x는 이미지당 3회 지연 측정",
        "- 이미지 경로·정답 순서를 RTMDet-x와 선택 모델 manifest 사이에서 검증",
        "", "## 해석 주의사항", "",
        "- Top-2는 최대 두 후보를 반환하므로 정답 포함률이 구조적으로 상승할 수 있습니다.",
        "- 자동 승인 정책에서는 FNR 감소와 함께 FPR 증가를 반드시 확인해야 합니다.",
        "- 모델 선택에 이미 사용한 평가 세트이므로 최종 정책과 임계값은 별도 검증 데이터에서 확정해야 합니다.",
        "", "산출물: [통합 CSV](comparison-with-rtmdet.csv) · [클래스별 CSV](class-metrics.csv) · [모델별 JSON](summary.json) · 모델별 `topk-predictions.csv`", "",
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Evaluation output containing run.json and model folders")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reports/selected-topk2")
    parser.add_argument(
        "--rtmdet-predictions",
        type=Path,
        default=ROOT / "outputs/rtmdet/model-size-20260927/rtmdet-x/predictions.csv",
    )
    parser.add_argument(
        "--rtmdet-metrics",
        type=Path,
        default=ROOT / "outputs/rtmdet/model-size-20260927/rtmdet-x/metrics.json",
    )
    args = parser.parse_args()
    args.input, args.output, args.rtmdet_predictions, args.rtmdet_metrics = (
        args.input.resolve(),
        args.output.resolve(),
        args.rtmdet_predictions.resolve(),
        args.rtmdet_metrics.resolve(),
    )
    manifest_path = args.input / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    threshold = float(manifest["protocol"]["threshold"])
    if tuple(manifest["models"]) != SELECTED_MODELS:
        raise ValueError(f"Expected exactly {SELECTED_MODELS}, got {tuple(manifest['models'])}")

    args.output.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict] = {}
    comparison_rows: list[dict] = []
    class_rows: list[dict] = []
    model_sources = {
        "rtmdet-x": (args.rtmdet_predictions, args.rtmdet_metrics),
        **{
            model: (args.input / model / "predictions.csv", args.input / model / "metrics.json")
            for model in SELECTED_MODELS
        },
    }
    for model in ALL_MODELS:
        predictions_path, metrics_path = model_sources[model]
        rows = read_csv(predictions_path)
        validate_predictions(model, rows, manifest)
        top1, top1_detail = summarize(rows, threshold, 1)
        top2, top2_detail = summarize(rows, threshold, 2)
        stored_predictions = [row["prediction"] for row in rows]
        calculated_top1 = [row["accepted_labels"] for row in top1_detail]
        if stored_predictions != calculated_top1:
            raise ValueError(f"{model} stored Top-1 decisions differ from recalculated decisions")

        details = []
        for first, second in zip(top1_detail, top2_detail):
            details.append({
                "path": first["path"], "sha256": first["sha256"], "true_label": first["true_label"],
                "top1_labels": first["accepted_labels"], "top1_contains_truth": first["contains_truth"],
                "top2_labels": second["accepted_labels"], "top2_contains_truth": second["contains_truth"],
                **{f"score_{group}": first[f"score_{group}"] for group in GROUPS},
            })
        model_output = args.output / model
        model_output.mkdir(exist_ok=True)
        write_csv(model_output / "topk-predictions.csv", details)
        write_json(model_output / "summary.json", {"model": model, "threshold": threshold, "top1": top1, "top2": top2})
        summaries[model] = {"top1": top1, "top2": top2}

        for summary in (top1, top2):
            for group in TARGET_GROUPS:
                class_rows.append(
                    {
                        "model": model,
                        "display_name": DISPLAY_NAMES[model],
                        "top_k": summary["top_k"],
                        "class": group,
                        **summary["class_metrics"][group],
                    }
                )

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        latency = (
            metrics["runtime"]["latency_mean_ms"]
            if model == "rtmdet-x"
            else metrics["mean_latency_ms"]
        )
        for label, summary in (("Top-1", top1), ("Top-2", top2)):
            comparison_rows.append({
                "model": model,
                "display_name": DISPLAY_NAMES[model],
                "strategy": label,
                "top_k": summary["top_k"],
                "images": summary["images"],
                "truth_inclusion_rate": summary["truth_inclusion_rate"],
                "fpr": summary["fpr"],
                "fnr": summary["fnr"],
                "expected_label_fnr": summary["expected_label_fnr"],
                "multiple_candidate_rate": summary["multiple_candidate_rate"],
                "latency_ms": latency,
            })

    write_csv(args.output / "comparison-with-rtmdet.csv", comparison_rows)
    write_csv(args.output / "class-metrics.csv", class_rows)
    setup_plotting()
    save_topk_overview(args.output, summaries)
    save_recovery_cost(args.output, summaries)
    save_tradeoff(args.output, comparison_rows)
    save_class_heatmaps(args.output, summaries)
    save_top1_reference(args.output, comparison_rows)
    write_json(
        args.output / "summary.json",
        {
            "models_compared": ALL_MODELS,
            "threshold": threshold,
            "inference_manifest_sha256": manifest["manifest_sha256"],
            "inference_run_sha256": sha256(manifest_path),
            "rtmdet_predictions_source": str(args.rtmdet_predictions),
            "rtmdet_predictions_sha256": sha256(args.rtmdet_predictions),
            "rtmdet_metrics_source": str(args.rtmdet_metrics),
            "rtmdet_metrics_sha256": sha256(args.rtmdet_metrics),
            "models": summaries,
        },
    )
    build_report(args.output, comparison_rows, summaries)
    print(args.output / "README.md")


if __name__ == "__main__":
    main()
