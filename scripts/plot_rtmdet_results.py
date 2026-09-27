"""Plot RTMDet Top-1 metrics from the archived 2026-09-27 confusion matrices."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/reports/rtmdet_top1_confusion_matrices.csv"
DESTINATION = ROOT / "docs/assets"
MODELS = ("tiny", "s", "m", "l", "x")
LABELS = ("computer", "book", "other")


def load_matrices() -> dict[str, np.ndarray]:
    matrices: dict[str, dict[str, list[int]]] = {model: {} for model in MODELS}
    with SOURCE.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            model = row["model"]
            label = row["true_label"]
            if model not in matrices or label not in LABELS or label in matrices[model]:
                raise ValueError(f"Unexpected or duplicate matrix row: {model}/{label}")
            matrices[model][label] = [int(row[name]) for name in LABELS]

    if any(set(rows) != set(LABELS) for rows in matrices.values()):
        raise ValueError("Every model must have computer, book, and other rows")
    return {
        model: np.asarray([rows[label] for label in LABELS], dtype=int)
        for model, rows in matrices.items()
    }


def rates(matrix: np.ndarray) -> tuple[float, float, float]:
    total = matrix.sum()
    accuracy = np.trace(matrix) / total
    # Binary MVP acceptance: computer/book = target; other = non-target.
    false_accepts = matrix[2, :2].sum()
    non_targets = matrix[2, :].sum()
    false_rejects = matrix[:2, 2].sum()
    targets = matrix[:2, :].sum()
    return accuracy * 100, false_accepts / non_targets * 100, false_rejects / targets * 100


def plot_comparison(matrices: dict[str, np.ndarray]) -> None:
    values = np.asarray([rates(matrices[model]) for model in MODELS])
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    specs = (
        ("Top-1 accuracy", 0, 90, "Goal ≥ 90%", "#3267a8"),
        ("MVP false positive rate", 1, 5, "Goal ≤ 5%", "#cf7541"),
        ("MVP false negative rate", 2, 5, "Goal ≤ 5%", "#b94b63"),
    )
    for axis, (title, column, target, target_label, color) in zip(axes, specs):
        bars = axis.bar(MODELS, values[:, column], color=color, width=0.63)
        axis.axhline(target, color="#303030", linestyle="--", linewidth=1.4)
        for bar, value in zip(bars, values[:, column]):
            label_y = value - 12 if column == 0 else value + 0.7
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{value:.2f}%",
                ha="center",
                fontsize=9,
                color="white" if column == 0 else "#1f2937",
            )
        axis.set_ylim(0, max(values[:, column].max(), target) * 1.25 + 1)
        axis.set_title(f"{title}\n{target_label}")
        axis.set_ylabel("Percent")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("RTMDet Top-1 on 372 images (computer, book, other)", fontsize=13)
    figure.savefig(DESTINATION / "rtmdet_top1_metrics.png", dpi=170)
    plt.close(figure)


def plot_confusion(matrix: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    figure.colorbar(image, ax=axis, label="Images", shrink=0.8)
    axis.set_xticks(range(3), LABELS)
    axis.set_yticks(range(3), LABELS)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("RTMDet-x Top-1 confusion matrix (n=372)")
    for row in range(3):
        for column in range(3):
            count = matrix[row, column]
            percent = count / matrix[row].sum() * 100
            color = "white" if count > matrix.max() / 2 else "#1f2937"
            axis.text(column, row, f"{count}\n({percent:.1f}% of row)", ha="center", va="center", color=color)
    figure.savefig(DESTINATION / "rtmdet_x_top1_confusion_matrix.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    matrices = load_matrices()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    plot_comparison(matrices)
    plot_confusion(matrices["x"])
