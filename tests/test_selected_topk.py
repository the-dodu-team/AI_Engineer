import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_selected_topk import accepted_groups, summarize


def prediction(true_label, computer, book, other, name="image.jpg"):
    return {
        "path": name,
        "true_label": true_label,
        "score_computer": str(computer),
        "score_book": str(book),
        "score_other": str(other),
    }


def test_top2_recovers_second_ranked_truth():
    rows = [prediction("book", .8, .7, .1)]
    top1, _ = summarize(rows, threshold=.05, top_k=1)
    top2, _ = summarize(rows, threshold=.05, top_k=2)
    assert top1["truth_inclusion_rate"] == 0
    assert top2["truth_inclusion_rate"] == 1
    assert top1["expected_label_fnr"] == 1
    assert top2["expected_label_fnr"] == 0


def test_top2_false_accept_counts_any_target_on_other_image():
    rows = [prediction("other", .7, .1, .8)]
    top1, _ = summarize(rows, threshold=.05, top_k=1)
    top2, _ = summarize(rows, threshold=.05, top_k=2)
    assert top1["fpr"] == 0
    assert top2["fpr"] == 1


def test_no_passing_score_rejects_as_other():
    assert accepted_groups({"computer": .01, "book": .02, "other": .03}, .05, 2) == ("other",)


def test_invalid_top_k_is_rejected():
    with pytest.raises(ValueError, match="top_k"):
        accepted_groups({"computer": .1, "book": .2, "other": .3}, .05, 0)
