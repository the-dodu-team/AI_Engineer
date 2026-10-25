"""CLI arguments shared by both stages, plus run metadata for reproducibility."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

from .backbones import MODELS
from .data import Record, count_table, dedupe, discover, limit_per_group, make_splits

DEFAULT_OUT_ROOT = Path("outputs/embed_verify")


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-dir", type=Path, required=True,
                        help="Folder with laptop/ and book/ sub-folders (e.g. the Drive sampleV2 folder).")
    parser.add_argument("--other-dir", type=Path, action="append", default=[],
                        help="Folder whose sub-folders are 'other' sub-classes (bed, table, ...). Repeatable. "
                             "Sub-folders named laptop/book are skipped automatically.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation share (threshold/model selection).")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test share (reported results only).")
    parser.add_argument("--other-in-val", action="store_true",
                        help="Share 'other' between validation (threshold tuning) and test. Default: 'other' is "
                             "used for testing only and thresholds come from laptop/book validation images.")
    parser.add_argument("--other-in-train", action="store_true",
                        help="Also train on 'other' images (implies --other-in-val). Default: off.")
    parser.add_argument("--unseen-other", nargs="*", default=[],
                        help="'Other' sub-class names kept out of training/validation and used only "
                             "for testing (open-set check).")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Split seeds.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch default)")
    parser.add_argument("--limit-per-group", type=int, default=None,
                        help="Smoke runs: keep at most N images per (label, subclass, source).")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", default=None, help="Output sub-folder name (default: <stage>-<timestamp>).")


def load_records(args) -> tuple[list[Record], dict]:
    train, bad_train = discover(args.train_dir, "train")
    groups, bad = [train], bad_train
    for other_dir in args.other_dir:
        recs, b = discover(other_dir, "other", force_other=True, skip_target_names=True)
        groups.append(recs)
        bad += b
    if not 0 < args.val_ratio < 1 or not 0 < args.test_ratio < 1 or args.val_ratio + args.test_ratio >= 0.9:
        raise SystemExit("--val-ratio and --test-ratio must be in (0, 1) and sum to less than 0.9")
    records, report = dedupe(groups)
    records = limit_per_group(records, args.limit_per_group)
    report["unreadable_skipped"] = bad
    labels = {r.label for r in records}
    if not {"laptop", "book"} <= labels:
        raise SystemExit(f"Need both laptop and book images under {args.train_dir}; found {sorted(labels)}")
    if args.unseen_other and not (args.other_in_val or args.other_in_train):
        print("Note: 'other' is test-only by default, so every 'other' kind is already unseen; "
              "--unseen-other is ignored.", flush=True)
        args.unseen_other = []
    missing = set(args.unseen_other) - {r.subclass for r in records if r.label == "other"}
    if missing:
        have = sorted({r.subclass for r in records if r.label == "other"})
        raise SystemExit(f"--unseen-other names not found: {sorted(missing)}. Available 'other' sub-classes: {have}")
    return records, report


def split_for_seed(records: list[Record], seed: int, args) -> list[str]:
    ratios = (1 - args.val_ratio - args.test_ratio, args.val_ratio, args.test_ratio)
    return make_splits(records, seed, frozenset(args.unseen_other), ratios, args.other_in_train, args.other_in_val)


def make_out_dir(args, stage: str) -> Path:
    name = args.run_name or f"{stage}-{datetime.now():%Y%m%d-%H%M%S}"
    out = args.out_root / name
    if (out / "metrics.csv").exists():
        raise SystemExit(f"{out} already has results; choose another --run-name")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _git(*cmd: str) -> str:
    try:
        return subprocess.run(["git", *cmd], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def write_run_config(out: Path, args, stage: str, records: list[Record], dedupe_report: dict,
                     extra: dict | None = None) -> None:
    import numpy
    import sklearn
    import transformers

    config = {
        "stage": stage, "created": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "data": count_table(records), "dedupe": dedupe_report,
        "git_commit": _git("rev-parse", "HEAD"), "git_branch": _git("branch", "--show-current"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "numpy": numpy.__version__, "sklearn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(), "torch_threads": torch.get_num_threads(),
        **(extra or {}),
    }
    (out / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    import csv
    with (out / "data_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "subclass", "source", "origin", "sha256"])
        for r in records:
            writer.writerow([r.path, r.label, r.subclass, r.source, r.origin, r.sha256])
