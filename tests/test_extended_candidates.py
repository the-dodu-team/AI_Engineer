import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_coco_candidates import decide
from evaluate_extended_candidates import canonical_label, validate_manifest
from compare_extended_candidates import read_predictions, summarize, paired_counts


def test_old_coco_tv_name_is_computer():
    assert decide([("tvmonitor", .9), ("book", .3)], .05)[0] == "computer"
    detections = [(canonical_label("tvmonitor"), .9), ("book", .3)]
    assert decide(detections, .05)[0] == "computer"


def test_same_count_different_image_is_rejected():
    baseline = {"images": [{"sha256": "old", "source_label": "bed"}], "smoke_test": False}
    validate_manifest([{"sha256": "old", "true_label": "other"}], baseline)
    with pytest.raises(ValueError, match="Dataset differs"):
        validate_manifest([{"sha256": "new", "true_label": "other"}], baseline)


def test_duplicate_prediction_cannot_pass_by_matching_count(tmp_path):
    expected = {str(tmp_path / "a.jpg").casefold(): {"true_label": "book", "sha256": "a"},
                str(tmp_path / "b.jpg").casefold(): {"true_label": "book", "sha256": "b"}}
    path = tmp_path / "predictions.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "true_label", "prediction"])
        writer.writeheader()
        writer.writerows([{"path": str(tmp_path / "a.jpg"), "true_label": "book", "prediction": "book"}]*2)
    with pytest.raises(ValueError, match="Duplicate"):
        read_predictions(path, expected)


def test_cross_target_error_is_accuracy_error_but_not_rejection():
    rows = {"a": {"true_label": "book", "prediction": "computer"},
            "b": {"true_label": "computer", "prediction": "other"},
            "c": {"true_label": "other", "prediction": "other"}}
    result = summarize(rows)
    assert result["accuracy"] == 1/3
    assert result["fnr"] == .5
    assert result["fpr"] == 0
    assert result["target_class_confusions"] == 1


def test_paired_counts_track_improvements_and_regressions():
    old = {"a": {"true_label": "book", "prediction": "other"},
           "b": {"true_label": "other", "prediction": "other"}}
    new = {"a": {"true_label": "book", "prediction": "book"},
           "b": {"true_label": "other", "prediction": "book"}}
    result = paired_counts(new, old)
    assert result["fixed_vs_rtmdet_x"] == result["regressed_vs_rtmdet_x"] == 1
    assert result["mcnemar_exact_p_unadjusted"] == 1
