"""Build the PM-facing RTMDet versus latest-detector comparison report."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COMBINED_CSV = ROOT / "docs/reports/coco-extended-detectors/comparison.csv"
RTMDET_CSV = ROOT / "outputs/rtmdet/model-size-20260927/model-size-comparison/comparison.csv"
OUTPUT = ROOT / "docs/reports/pm-rtmdet-vs-latest"

RTMDET_MODELS = ["rtmdet-tiny", "rtmdet-s", "rtmdet-m", "rtmdet-l", "rtmdet-x"]
LATEST_MODELS = [
    "rf-detr-large",
    "dfine-l-obj2coco",
    "lw-detr-large",
    "rf-detr-medium",
    "yolov13l",
]
DECISION_MODELS = ["rtdetrv2-r50", *LATEST_MODELS, *RTMDET_MODELS]

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
AMBER = "#F59E0B"
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
    if model == "rtdetrv2-r50":
        return BLUE
    if model in RTMDET_MODELS:
        return SLATE
    return AMBER


def save_overview(rows: dict[str, dict[str, str]]) -> None:
    ordered = sorted(DECISION_MODELS, key=lambda model: float(rows[model]["accuracy"]))
    names = [DISPLAY[m] for m in ordered]
    accuracies = [pct(rows[m]["accuracy"]) for m in ordered]
    fnrs = [pct(rows[m]["fnr"]) for m in ordered]
    fprs = [pct(rows[m]["fpr"]) for m in ordered]
    deltas = [int(rows[m]["correct"]) - int(rows["rtmdet-x"]["correct"]) for m in ordered]

    fig = plt.figure(figsize=(16, 12), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1])
    ax_accuracy = fig.add_subplot(grid[0, :])
    ax_error = fig.add_subplot(grid[1, 0])
    ax_delta = fig.add_subplot(grid[1, 1])

    fig.suptitle("RTMDet 모델 크기별 결과 vs 최신 사전학습 후보", fontsize=23, fontweight="bold")
    fig.text(
        0.5,
        0.955,
        "동일한 372개 이미지 · Top-1 분류 기준 | 파랑: 추천 · 주황: 최신 후보 · 회색: RTMDet",
        ha="center",
        fontsize=12,
        color="#475569",
    )

    bars = ax_accuracy.barh(names, accuracies, color=[color_for(m) for m in ordered], height=0.68)
    ax_accuracy.set_xlim(85, 97)
    ax_accuracy.set_xlabel("정확도 (%)")
    ax_accuracy.set_title("A. 정확도 — RT-DETRv2-R50이 1위", loc="left", fontsize=15, fontweight="bold")
    ax_accuracy.grid(axis="x", color=GRID, linewidth=0.8)
    ax_accuracy.set_axisbelow(True)
    for bar, value in zip(bars, accuracies):
        ax_accuracy.text(value + 0.12, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center", fontweight="bold", fontsize=10)
    for spine in ax_accuracy.spines.values():
        spine.set_visible(False)

    y = np.arange(len(names))
    width = 0.38
    ax_error.barh(y - width / 2, fnrs, height=width, color=RED, label="FNR: 대상 누락")
    ax_error.barh(y + width / 2, fprs, height=width, color=GREEN, label="FPR: 비대상 오수락")
    ax_error.axvline(5, color=NAVY, linestyle="--", linewidth=1.5, label="목표 5%")
    ax_error.set_yticks(y, names)
    ax_error.set_xlim(0, 17)
    ax_error.set_xlabel("오류율 (%) — 낮을수록 좋음")
    ax_error.set_title("B. 운영 오류 — 모든 모델이 FNR 목표 미달", loc="left", fontsize=15, fontweight="bold")
    ax_error.grid(axis="x", color=GRID, linewidth=0.8)
    ax_error.set_axisbelow(True)
    ax_error.legend(loc="lower right", frameon=False, fontsize=9)
    for spine in ax_error.spines.values():
        spine.set_visible(False)

    delta_colors = [BLUE if m == "rtdetrv2-r50" else (GREEN if d > 0 else RED if d < 0 else SLATE) for m, d in zip(ordered, deltas)]
    delta_bars = ax_delta.barh(names, deltas, color=delta_colors, height=0.68)
    ax_delta.axvline(0, color=NAVY, linewidth=1.2)
    ax_delta.set_xlim(-26, 8)
    ax_delta.set_xlabel("RTMDet-x 대비 추가 정답 수 (장)")
    ax_delta.set_title("C. RTMDet-x 대비 — 추천 모델 +6장", loc="left", fontsize=15, fontweight="bold")
    ax_delta.grid(axis="x", color=GRID, linewidth=0.8)
    ax_delta.set_axisbelow(True)
    for bar, value in zip(delta_bars, deltas):
        x = value + 0.35 if value >= 0 else value - 0.35
        align = "left" if value >= 0 else "right"
        ax_delta.text(x, bar.get_y() + bar.get_height() / 2, f"{value:+d}", va="center", ha=align, fontweight="bold")
    for spine in ax_delta.spines.values():
        spine.set_visible(False)

    fig.savefig(OUTPUT / "pm-overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_rtmdet_trend(rows: list[dict[str, str]]) -> None:
    top1 = {row["model"]: row for row in rows if row["top_k"] == "1"}
    x = np.arange(len(RTMDET_MODELS))
    accuracy = [pct(top1[m]["candidate_hit_rate"]) for m in RTMDET_MODELS]
    latency = [float(top1[m]["latency_mean_ms"]) for m in RTMDET_MODELS]

    fig, ax_accuracy = plt.subplots(figsize=(13, 5.8), layout="constrained")
    ax_latency = ax_accuracy.twinx()
    ax_accuracy.plot(x, accuracy, color=BLUE, marker="o", linewidth=3, markersize=9, label="정확도")
    ax_latency.plot(x, latency, color=AMBER, marker="o", linewidth=3, markersize=9, label="CPU 지연")
    ax_accuracy.set_xticks(x, [DISPLAY[m] for m in RTMDET_MODELS])
    ax_accuracy.set_ylim(85, 96)
    ax_latency.set_ylim(0, 1500)
    ax_accuracy.set_ylabel("정확도 (%)", color=BLUE, fontweight="bold")
    ax_latency.set_ylabel("지연 (ms/image)", color=AMBER, fontweight="bold")
    ax_accuracy.grid(axis="y", color=GRID, linewidth=0.8)
    ax_accuracy.set_axisbelow(True)
    ax_accuracy.set_title("RTMDet 크기 증가 효과: x가 가장 정확하지만 지연은 tiny의 7.7배", loc="left", fontsize=16, fontweight="bold")
    for i, value in enumerate(accuracy):
        ax_accuracy.annotate(f"{value:.2f}%", (i, value), xytext=(0, 18), textcoords="offset points", ha="center", color=BLUE, fontweight="bold")
    for i, value in enumerate(latency):
        ax_latency.annotate(f"{value:.0f}ms", (i, value), xytext=(0, -28), textcoords="offset points", ha="center", color="#B45309", fontweight="bold")
    for axis in (ax_accuracy, ax_latency):
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.savefig(OUTPUT / "rtmdet-size-trend.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_readme(rows: dict[str, dict[str, str]]) -> None:
    baseline = rows["rtmdet-x"]
    recommendation = rows["rtdetrv2-r50"]
    selected = sorted(DECISION_MODELS, key=lambda model: float(rows[model]["accuracy"]), reverse=True)

    lines = [
        "# PM 보고: RTMDet 모델 크기별 결과 vs 최신 탐지 모델",
        "",
        "> **제안: RT-DETRv2-R50을 다음 PoC 모델로 채택하고, RTMDet-x는 기준 모델로 유지합니다.**",
        "> 정확도는 95.70%로 RTMDet-x보다 6장을 더 맞혔지만, FNR 5% 목표와 독립 검증은 아직 통과하지 못했습니다.",
        "",
        "![RTMDet와 최신 모델 비교 대시보드](pm-overview.png)",
        "",
        "## 한눈에 보는 의사결정",
        "",
        "| 질문 | 답변 | 판단 |",
        "|---|---|---|",
        f"| RTMDet보다 정확한 모델이 있는가? | **있음. RT-DETRv2-R50 {pct(recommendation['accuracy']):.2f}%** vs RTMDet-x {pct(baseline['accuracy']):.2f}% | 다음 검증 진행 |",
        f"| 개선폭은 충분한가? | **+6장, +{pct(recommendation['accuracy']) - pct(baseline['accuracy']):.2f}%p** | 방향은 긍정적이나 통계적으로 미확정 |",
        f"| 대상 누락이 줄었는가? | FNR **{pct(baseline['fnr']):.2f}% → {pct(recommendation['fnr']):.2f}%** | 21장 → 15장으로 개선 |",
        f"| 비대상 오수락은 늘었는가? | FPR **{pct(baseline['fpr']):.2f}% → {pct(recommendation['fpr']):.2f}%** | 동일, 각 1장 |",
        "| 상업 적용이 가능한가? | RT-DETRv2-R50·RTMDet은 Apache-2.0 계열 | 고지 조건을 지키면 유리 |",
        "| 바로 제품에 적용할 수 있는가? | **아직 아님** | 독립 데이터·임계값·실서비스 속도 검증 필요 |",
        "",
        "## RTMDet 크기별 결과",
        "",
        "RTMDet은 전반적으로 모델이 커질수록 정확도가 좋아졌지만, `m → l`에서는 정확도가 오히려 0.54%p 낮아졌습니다. 가장 정확한 `x`는 `tiny`보다 정답이 24장 많고 CPU 지연은 약 7.7배입니다.",
        "",
        "![RTMDet 모델 크기별 정확도와 지연](rtmdet-size-trend.png)",
        "",
        "| 모델 | 정답/372 | 정확도 | FPR | FNR | CPU 지연¹ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in RTMDET_MODELS:
        row = rows[model]
        lines.append(
            f"| {DISPLAY[model]} | {row['correct']} | {pct(row['accuracy']):.2f}% | {pct(row['fpr']):.2f}% | {pct(row['fnr']):.2f}% | {float(row['latency_ms']):.0f} ms |"
        )

    lines += [
        "",
        "## 최신 후보와 직접 비교",
        "",
        "`최신 후보`는 이번에 추가 실행한 5개 모델입니다. RT-DETRv2-R50은 앞선 후보 평가의 클래스 매핑 오류를 보정한 결과이며, 전체 모델 중 성능이 가장 좋아 의사결정 표에 함께 넣었습니다.",
        "",
        "| 순위 | 모델 | 구분 | 정답/372 | 정확도 | FPR | FNR | RTMDet-x 대비 | 상업 적용 검토 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for rank, model in enumerate(selected, 1):
        row = rows[model]
        group = "추천" if model == "rtdetrv2-r50" else "RTMDet" if model in RTMDET_MODELS else "최신 후보"
        delta = int(row["correct"]) - int(baseline["correct"])
        commercial = {
            "permissive_with_notices": "가능성 높음",
            "objects365_terms_review": "권리 확인 필요",
            "copyleft_or_separate_permission": "AGPL/별도 허가",
        }.get(row["commercial_status"], "확인 필요")
        lines.append(
            f"| {rank} | **{DISPLAY[model]}** | {group} | {row['correct']} | {pct(row['accuracy']):.2f}% | {pct(row['fpr']):.2f}% | {pct(row['fnr']):.2f}% | {delta:+d}장 | {commercial} |"
        )

    lines += [
        "",
        "## PM 권고안",
        "",
        "### 1. RT-DETRv2-R50으로 제한된 PoC 진행",
        "",
        "- 현재 표본에서 정확도 1위이며 RTMDet-x와 같은 FPR에서 FNR을 2.21%p 낮췄습니다.",
        "- Apache-2.0 계열이라 폐쇄형 상업 서비스 검토가 상대적으로 단순합니다.",
        "- 파라미터는 43.0M으로 현재 프로젝트의 `RTMDet-l보다 작은 후보` 조건을 충족합니다.",
        "",
        "### 2. RTMDet-x는 비교 기준으로 유지",
        "",
        "- 최신 후보 5개 중 D-FINE-L만 RTMDet-x보다 2장 더 맞혔습니다.",
        "- RF-DETR Large와 YOLOv13-L은 동률, RF-DETR Medium과 LW-DETR Large는 더 낮았습니다.",
        "- RTMDet-x는 라이선스가 명확하고 기존 구현이 있어 롤백 기준으로 적합합니다.",
        "",
        "### 3. 제품 전환 전에 세 가지 게이트 적용",
        "",
        "1. **독립 검증:** 모델 선정에 사용하지 않은 신규 이미지로 재평가합니다.",
        "2. **운영 목표:** 검증 세트에서 FPR ≤ 5%, FNR ≤ 5%를 모두 통과해야 합니다. 현재 최선 FNR은 5.54%로 누락 2장을 더 줄여야 합니다.",
        "3. **배포 검증:** 같은 서버·런타임·반복 횟수로 GPU 지연과 메모리를 다시 측정합니다.",
        "",
        "## 해석 시 주의사항",
        "",
        "- 결과는 고유 이미지 372장(`computer` 200, `book` 71, `other` 101)의 이미지 단위 Top-1 평가입니다.",
        "- RT-DETRv2-R50과 RTMDet-x의 이미지별 차이는 exact McNemar `p=0.238`입니다. 현재 표본만으로 우위를 확정할 수 없습니다.",
        "- ¹ RTMDet 5종의 지연은 같은 기존 실행 안에서 비교할 수 있습니다. 최신 후보는 프레임워크와 반복 횟수가 달라 RTMDet과의 속도 배수를 보고하지 않습니다.",
        "- D-FINE-L과 LW-DETR은 Objects365 관련 가중치 권리를 확인해야 합니다. YOLOv13-L은 AGPL-3.0 준수 또는 별도 허가가 필요합니다.",
        "",
        "## 근거 자료",
        "",
        "- [20개 모델 전체 집계](../coco-extended-detectors/README.md)",
        "- [통합 수치 CSV](../coco-extended-detectors/comparison.csv)",
        "- [기존 RTMDet 크기별 원본 결과](../../../outputs/rtmdet/model-size-20260927/model-size-comparison/README.md)",
        "- [평가 및 재현 방법](../../coco-extended-detectors.md)",
        "",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    combined = {row["model"]: row for row in read_rows(COMBINED_CSV)}
    missing = set(DECISION_MODELS) - set(combined)
    if missing:
        raise RuntimeError(f"Missing models in {COMBINED_CSV}: {sorted(missing)}")
    rtmdet = read_rows(RTMDET_CSV)
    setup_plotting()
    save_overview(combined)
    save_rtmdet_trend(rtmdet)
    build_readme(combined)
    print(OUTPUT / "README.md")


if __name__ == "__main__":
    main()
