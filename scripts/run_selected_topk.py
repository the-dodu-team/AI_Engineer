"""Run the three selected detectors once, then build Top-1/Top-2 reports."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SELECTED_MODELS = ("rtdetrv2-r50", "rtdetrv2-r34", "lw-detr-large")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/selected-topk2")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/reports/selected-topk2")
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/rtmdet/model-size-20260927/run.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "models/temporary/selected-topk2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-weights", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    output, report = args.output.resolve(), args.report.resolve()
    if not args.report_only:
        if args.source is None:
            parser.error("--source is required unless --report-only is used")
        command = [
            sys.executable,
            str(ROOT / "scripts/evaluate_extended_candidates.py"),
            "--source", str(args.source.resolve()),
            "--output", str(output),
            "--baseline", str(args.baseline.resolve()),
            "--cache-dir", str(args.cache_dir.resolve()),
            "--models", *SELECTED_MODELS,
            "--device", args.device,
            "--threshold", str(args.threshold),
            "--warmup", str(args.warmup),
            "--repeats", str(args.repeats),
            "--threads", str(args.threads),
        ]
        if args.resume:
            command.append("--resume")
        if args.keep_weights:
            command.append("--keep-weights")
        subprocess.run(command, cwd=ROOT, check=True)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/compare_selected_topk.py"),
            "--input", str(output),
            "--output", str(report),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
