"""Compare Top-1 and Top-2 for the three selected detectors.

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
DISPLAY_NAMES = {
    "rtdetrv2-r50": "RT-DETRv2-R50",
    "rtdetrv2-r34": "RT-DETRv2-R34",
    "lw-detr-large": "LW-DETR Large",
    "rtmdet-tiny": "RTMDet-tiny",
    "rtmdet-s": "RTMDet-s",
    "rtmdet-m": "RTMDet-m",
    "rtmdet-l": "RTMDet-l",
    "rtmdet-x": "RTMDet-x",
}
SHORT_NAMES = {
    "rtdetrv2-r50": "RT-DETRv2-R50",
    "rtdetrv2-r34": "RT-DETRv2-R34",
    "lw-detr-large": "LW-DETR Large",
    "rtmdet-tiny": "RTMDet-tiny",
    "rtmdet-s": "RTMDet-s",
    "rtmdet-m": "RTMDet-m",
    "rtmdet-l": "RTMDet-l",
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
        accepted_count = tp + fp
        class_metrics[group] = {
            "support": len(actual_positive),
            "tp": tp,
            "fn": len(actual_positive) - tp,
            "fp": fp,
            "recall": tp / len(actual_positive) if actual_positive else 0.0,
            "fnr": 1 - tp / len(actual_positive) if actual_positive else 0.0,
            "precision": tp / accepted_count if accepted_count else 0.0,
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


def historical_rtmdet_rows(path: Path) -> list[dict]:
    rows = read_csv(path)
    selected = [row for row in rows if row["model"].startswith("rtmdet-")]
    if not selected:
        raise ValueError(f"No RTMDet rows in {path}")
    return [
        {
            "model": row["model"],
            "display_name": DISPLAY_NAMES.get(row["model"], row["model"]),
            "strategy": "historical Top-1",
            "top_k": 1,
            "images": int(row["images"]),
            "truth_inclusion_rate": float(row["accuracy"]),
            "fpr": float(row["fpr"]),
            "fnr": float(row["fnr"]),
            "expected_label_fnr": "",
            "multiple_candidate_rate": 0.0,
            "latency_ms": float(row["latency_ms"]),
        }
        for row in selected
    ]


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
    names = [SHORT_NAMES[model] for model in SELECTED_MODELS]
    x = np.arange(len(names))
    width = 0.34
    panels = [
        ("truth_inclusion_rate", "Truth inclusion", (90, 101), GREEN),
        ("fnr", "Target FNR", (0, 10), RED),
        ("fpr", "Target FPR", (0, 72), RED),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5), layout="constrained")
    for axis, (metric, title, limits, top2_color) in zip(axes, panels):
        top1 = [summaries[model]["top1"][metric] * 100 for model in SELECTED_MODELS]
        top2 = [summaries[model]["top2"][metric] * 100 for model in SELECTED_MODELS]
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
    fig.suptitle("Selected Detectors: Top-1 vs Top-2", fontsize=22, fontweight="bold")
    fig.savefig(output / "topk-overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_recovery_cost(output: Path, summaries: dict[str, dict]) -> None:
    names = [SHORT_NAMES[model] for model in SELECTED_MODELS]
    recovered = []
    added_false_accepts = []
    for model in SELECTED_MODELS:
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
            historical = row["strategy"].startswith("historical")
            top2 = row["strategy"] == "selected Top-2"
            color = SLATE if historical else RED if top2 else BLUE
            marker = "s" if historical else "^" if top2 else "o"
            axis.scatter(x, y, s=85, color=color, marker=marker, edgecolor="white", linewidth=.8, zorder=3)
            label = SHORT_NAMES.get(row["model"], row["model"])
            if not historical:
                label += " k2" if top2 else " k1"
            axis.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
        axis.axvline(5, color=GREEN, linestyle="--", linewidth=1.2)
        axis.axhline(5, color=GREEN, linestyle="--", linewidth=1.2)
        axis.fill_between([0, 5], 0, 5, color=GREEN, alpha=.08)
        axis.set_xlabel("FPR (%) — lower is better")
        axis.set_ylabel("FNR (%) — lower is better")
        axis.grid(color=GRID, linewidth=.7)
        axis.set_axisbelow(True)
        axis.set_title("Full range" if not zoom else "Top-1 / RTMDet zoom", loc="left", fontweight="bold")
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
    metric_rows = {metric: [] for metric in ("precision", "recall", "fnr", "fpr")}
    for model in SELECTED_MODELS:
        for key in ("top1", "top2"):
            summary = summaries[model][key]
            row_labels.append(f"{SHORT_NAMES[model]} k{summary['top_k']}")
            for metric in metric_rows:
                metric_rows[metric].append(
                    [summary["class_metrics"][group][metric] * 100 for group in TARGET_GROUPS]
                )
    panels = [
        ("precision", "Precision", "YlGnBu", 45, 100),
        ("recall", "Recall", "YlGnBu", 75, 100),
        ("fnr", "FNR", "YlOrRd", 0, 20),
        ("fpr", "FPR", "YlOrRd", 0, 25),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 12), layout="constrained")
    for axis, (metric, title, cmap, vmin, vmax) in zip(axes.flat, panels):
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
    fig.suptitle("Class-Level Top-k Metrics", fontsize=21, fontweight="bold")
    fig.savefig(output / "class-metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_top1_reference(output: Path, comparison_rows: list[dict]) -> None:
    rows = [row for row in comparison_rows if row["strategy"] != "selected Top-2"]
    rows.sort(key=lambda row: float(row["truth_inclusion_rate"]))
    names = [SHORT_NAMES.get(row["model"], row["model"]) for row in rows]
    colors = [BLUE if row["strategy"] == "selected Top-1" else SLATE for row in rows]
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
    fig.suptitle("Selected Top-1 vs Historical RTMDet", fontsize=21, fontweight="bold")
    fig.savefig(output / "top1-vs-rtmdet.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(output: Path, comparison_rows: list[dict], selected_summaries: dict[str, dict]) -> None:
    selected_rows = [row for row in comparison_rows if row["strategy"].startswith("selected")]
    rtmdet_rows = [row for row in comparison_rows if row["strategy"].startswith("historical")]
    lines = [
        "# 선택 모델 Top-1·Top-2 비교",
        "",
        "RT-DETRv2-R50, RT-DETRv2-R34, LW-DETR Large의 동일한 추론 점수에서 Top-1과 Top-2를 비교합니다.",
        "이 실험은 사전학습 모델의 추론 후처리 비교이며 fine-tuning이나 가중치 갱신이 아닙니다.",
        "",
        "## 결론",
        "",
        "- **자동 단일 판정은 RT-DETRv2-R50 Top-1이 가장 안정적**입니다: 정확도 95.70%, FNR 5.54%, FPR 0.99%.",
        "- Top-2는 세 모델 모두 전체 FNR을 0%로 낮췄지만 FPR이 33.66~65.35%로 증가했습니다.",
        "- Top-2 중에서는 **LW-DETR Large가 가장 낮은 FPR 33.66%**였지만 자동 승인에는 여전히 높습니다.",
        "- 따라서 Top-2는 자동 판정보다 후속 분류기 또는 사용자 확인에 전달하는 recall 우선 shortlist로 사용하는 편이 적절합니다.",
        "",
        "![Top-1과 Top-2 핵심 지표](topk-overview.png)",
        "",
        "## 선택 모델 결과",
        "",
        "전체 FNR은 대상 이미지에서 `computer` 또는 `book` 후보가 하나도 없는 비율이고, FPR은 `other` 이미지에 대상 후보가 하나 이상 포함된 비율입니다.",
        "정답 포함률은 폴더 정답이 후보 집합 안에 존재하는 비율이므로 Top-2 단일 분류 정확도로 해석하면 안 됩니다.",
        "",
        "| 모델 | k | 정답 포함률 | 전체 FNR | 전체 FPR | 정답 클래스 FNR | 복수 후보율 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected_rows:
        lines.append(
            f"| {row['display_name']} | {row['top_k']} | {pct(row['truth_inclusion_rate'])} | "
            f"{pct(row['fnr'])} | {pct(row['fpr'])} | {pct(row['expected_label_fnr'])} | "
            f"{pct(row['multiple_candidate_rate'])} |"
        )

    lines += [
        "",
        "## Top-2 변화",
        "",
        "| 모델 | 추가 정답 포함 | FNR 변화 | FPR 변화 |",
        "|---|---:|---:|---:|",
    ]
    for model in SELECTED_MODELS:
        top1, top2 = selected_summaries[model]["top1"], selected_summaries[model]["top2"]
        lines.append(
            f"| {DISPLAY_NAMES[model]} | +{top2['truth_included'] - top1['truth_included']} | "
            f"{top1['fnr']:.2%} → {top2['fnr']:.2%} | {top1['fpr']:.2%} → {top2['fpr']:.2%} |"
        )

    lines += [
        "",
        "![Top-2 복구량과 오수락 비용](recovery-vs-cost.png)",
        "",
        "## 등록 클래스별 지표",
        "",
        "| 모델 | k | 클래스 | Precision | Recall | FNR | FPR |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for model in SELECTED_MODELS:
        for key in ("top1", "top2"):
            summary = selected_summaries[model][key]
            for group in TARGET_GROUPS:
                item = summary["class_metrics"][group]
                lines.append(
                    f"| {DISPLAY_NAMES[model]} | {summary['top_k']} | {group} | "
                    f"{item['precision']:.2%} | {item['recall']:.2%} | "
                    f"{item['fnr']:.2%} | {item['fpr']:.2%} |"
                )

    lines += [
        "",
        "![클래스별 Top-k 지표](class-metrics.png)",
        "",
        "## FNR–FPR 트레이드오프",
        "",
        "왼쪽은 Top-2까지 포함한 전체 범위, 오른쪽은 Top-1과 RTMDet 결과를 확대한 그림입니다. 초록색 영역은 FNR·FPR이 모두 5% 이하인 목표 구간입니다.",
        "",
        "![FNR과 FPR 트레이드오프](fnr-fpr-tradeoff.png)",
        "",
        "## 기존 RTMDet Top-1 기준",
        "",
        "아래 수치는 현재 브랜치의 기존 372장 RTMDet 결과입니다. 선택 모델 실행은 같은 RTMDet 이미지 SHA-256 manifest와 일치할 때만 진행됩니다.",
        "",
        "| 모델 | 정확도 | FNR | FPR | CPU 지연 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rtmdet_rows, key=lambda item: float(item["truth_inclusion_rate"]), reverse=True):
        lines.append(
            f"| {row['display_name']} | {pct(row['truth_inclusion_rate'])} | {pct(row['fnr'])} | "
            f"{pct(row['fpr'])} | {row['latency_ms']:.1f} ms |"
        )

    lines += [
        "",
        "![선택 모델 Top-1과 RTMDet 비교](top1-vs-rtmdet.png)",
        "",
        "## 평가 조건",
        "",
        "- 고유 이미지 372장: computer 200, book 71, other 101",
        "- 그룹 임계값 0.05, 원시 탐지 하한 0.01, 입력 640 또는 체크포인트 기본 전처리",
        "- CPU 10 threads, warm-up 5회, 이미지당 측정 1회",
        "- 동일 이미지 manifest SHA-256을 기존 RTMDet 실행과 대조",
        "",
        "## 해석 주의사항",
        "",
        "- Top-2는 최대 두 후보를 반환하므로 정답 포함률과 recall은 구조적으로 상승할 수 있습니다.",
        "- 자동 승인 정책에서는 FPR과 precision 악화 여부를 우선 확인해야 합니다.",
        "- `정답 클래스 FNR`은 book을 computer로만 포함한 경우도 누락으로 계산하며, 전체 FNR보다 엄격한 지표입니다.",
        "- 모델 선택에 이미 사용한 372장이므로 최종 정책과 임계값은 별도 검증 데이터에서 확정해야 합니다.",
        "",
        "산출물: [통합 CSV](comparison-with-rtmdet.csv) · [클래스별 CSV](class-metrics.csv) · [모델별 JSON](summary.json) · 모델별 `topk-predictions.csv`",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Evaluation output containing run.json and model folders")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reports/selected-topk2")
    parser.add_argument(
        "--rtmdet-comparison",
        type=Path,
        default=ROOT / "docs/reports/coco-extended-detectors/comparison.csv",
    )
    args = parser.parse_args()
    args.input, args.output, args.rtmdet_comparison = (
        args.input.resolve(), args.output.resolve(), args.rtmdet_comparison.resolve()
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
    for model in SELECTED_MODELS:
        predictions_path = args.input / model / "predictions.csv"
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

        latency = json.loads((args.input / model / "metrics.json").read_text(encoding="utf-8"))["mean_latency_ms"]
        for label, summary in (("selected Top-1", top1), ("selected Top-2", top2)):
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

    comparison_rows.extend(historical_rtmdet_rows(args.rtmdet_comparison))
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
            "selected_models": SELECTED_MODELS,
            "threshold": threshold,
            "inference_manifest_sha256": manifest["manifest_sha256"],
            "inference_run_sha256": sha256(manifest_path),
            "historical_rtmdet_source": str(args.rtmdet_comparison),
            "historical_rtmdet_sha256": sha256(args.rtmdet_comparison),
            "models": summaries,
        },
    )
    build_report(args.output, comparison_rows, summaries)
    print(args.output / "README.md")


if __name__ == "__main__":
    main()
