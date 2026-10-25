"""Stage 1: frozen backbone + prototype / logistic-regression / MLP heads.

    python -m embed_verify.stage1_frozen --train-dir <laptop+book folder> --other-dir <other folder>
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .backbones import Backbone, cached_extract, measure_latency
from .data import count_table
from .evalcore import Recorder
from .heads import HEADS, l2_normalize
from .report import build_report
from .runinfo import add_data_args, load_records, make_out_dir, split_for_seed, write_run_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_args(parser)
    parser.add_argument("--methods", nargs="+", default=list(HEADS), choices=list(HEADS))
    parser.add_argument("--dinov2-pool", default="cls_mean", choices=["cls", "cls_mean"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-latency", action="store_true")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    records, dedupe_report = load_records(args)
    out = make_out_dir(args, "stage1")
    unseen = frozenset(args.unseen_other)
    y = np.array([r.y for r in records])
    paths, shas = [r.path for r in records], [r.sha256 for r in records]
    split_by_seed = {seed: split_for_seed(records, seed, args) for seed in args.seeds}
    write_run_config(out, args, "stage1", records, dedupe_report, {
        "split_counts": {str(s): count_table(records, sp)["split"] for s, sp in split_by_seed.items()}})
    with (out / "splits.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "path", "split"])
        for seed, splits in split_by_seed.items():
            writer.writerows([[seed, p, s] for p, s in zip(paths, splits)])
    print(f"{len(records)} images: {count_table(records)['label']} -> {out}", flush=True)

    recorder, latency_rows = Recorder(unseen), []
    cache_dir = args.out_root / "cache"
    for model_name in args.models:
        print(f"\n== {model_name}", flush=True)
        backbone = Backbone(model_name, args.device, args.dinov2_pool)
        X, hit = cached_extract(backbone, paths, shas, cache_dir, args.batch_size)
        print(f"embeddings {X.shape} ({'cache' if hit else 'computed'})", flush=True)
        if not args.no_latency:
            latency_rows.append(measure_latency(backbone, paths))
        del backbone
        X = l2_normalize(X)
        for seed in args.seeds:
            splits = np.array(split_by_seed[seed])
            tr, va = splits == "train", splits == "val"
            for method in args.methods:
                scorer = HEADS[method](X[tr], y[tr], X[va], y[va], seed=seed)
                recorder.add(stage="stage1", model=model_name, method=method, seed=seed,
                             S=scorer(X), records=records, splits=list(splits))
                print(f"  seed {seed} {method} done", flush=True)
        recorder.write(out)  # keep partial results if a later model fails
    if latency_rows:
        with (out / "latency.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(latency_rows[0]))
            writer.writeheader()
            writer.writerows(latency_rows)
    build_report([out])
    print(f"\nDone. Results: {out}\nGive {out / 'REPORT.md'} to Claude for the write-up.")


if __name__ == "__main__":
    main()
