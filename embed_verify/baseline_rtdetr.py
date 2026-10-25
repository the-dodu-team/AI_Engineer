"""Zero-shot RT-DETRv2-R50 baseline on the same train/val/test splits as stage 1.

The COCO-pretrained detector is not trained at all. Its score for a photo is the highest
detection confidence of the laptop group (laptop + tv, as in the earlier detector benchmark)
and of the book class, so the metrics line up with the embedding methods:

    python -m embed_verify.baseline_rtdetr --train-dir <sampleV2> --seeds 0 1 2 \
        --compare-with outputs/embed_verify/stage1-pos
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from .backbones import load_image
from .data import count_table
from .evalcore import Recorder
from .report import build_report
from .runinfo import add_data_args, load_records, make_out_dir, split_for_seed, write_run_config

REPO = "PekingU/rtdetr_v2_r50vd"
REVISION = "282494075698cab9faa1096ae26856890030c817"
MODEL_NAME = "rtdetrv2-r50"
METHOD = "zeroshot-detector"                # laptop/book scores only
METHOD_FULL = "zeroshot-detector-full"      # earlier rule: other-object detections compete too
DETECTION_FLOOR = 0.01       # raw detections below this are dropped (as in the earlier benchmark)
PREVIOUS_THRESHOLD = 0.05    # group threshold used by the earlier detector benchmark and API


def canonical(label: str) -> str:
    label = label.strip().lower()
    return "tv" if label in {"tvmonitor", "television", "tv monitor"} else label


class Detector:
    def __init__(self, device: str, laptop_labels: list[str]):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.device = device
        self.laptop_labels = {canonical(x) for x in laptop_labels}
        self.processor = AutoImageProcessor.from_pretrained(REPO, revision=REVISION)
        self.model = AutoModelForObjectDetection.from_pretrained(REPO, revision=REVISION).to(device).eval()
        self.id2label = self.model.config.id2label
        if not {"laptop", "book"} <= {canonical(v) for v in self.id2label.values()}:
            raise ValueError("Detector vocabulary has no laptop/book classes")

    @torch.inference_mode()
    def scores(self, paths: list[str]) -> np.ndarray:
        """[n, 3] highest detection confidence for (laptop group, book, any other COCO object)."""
        images = [load_image(p) for p in paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=[(im.height, im.width) for im in images], threshold=DETECTION_FLOOR)
        out = np.zeros((len(images), 3), dtype=np.float32)
        for i, res in enumerate(results):
            for label, score in zip(res["labels"].tolist(), res["scores"].tolist()):
                name = canonical(self.id2label[int(label)])
                if name in self.laptop_labels:
                    out[i, 0] = max(out[i, 0], score)
                elif name == "book":
                    out[i, 1] = max(out[i, 1], score)
                else:
                    out[i, 2] = max(out[i, 2], score)
        return out


def latency(detector: Detector, paths: list[str], n: int = 20, warmup: int = 3) -> dict:
    sample = paths[:n]
    order = [sample[i % len(sample)] for i in range(warmup + len(sample))]
    prep, fwd = [], []
    with torch.inference_mode():
        for i, path in enumerate(order):
            t0 = time.perf_counter()
            image = load_image(path)
            inputs = detector.processor(images=[image], return_tensors="pt").to(detector.device)
            t1 = time.perf_counter()
            outputs = detector.model(**inputs)
            detector.processor.post_process_object_detection(
                outputs, target_sizes=[(image.height, image.width)], threshold=DETECTION_FLOOR)
            t2 = time.perf_counter()
            if i >= warmup:
                prep.append((t1 - t0) * 1000)
                fwd.append((t2 - t1) * 1000)
    return {"model": MODEL_NAME, "hf_id": REPO, "revision": REVISION,
            "params_m": round(sum(p.numel() for p in detector.model.parameters()) / 1e6, 2),
            "embedding_dim": "-", "device": detector.device, "threads": torch.get_num_threads(),
            "preprocess_ms_median": round(statistics.median(prep), 2),
            "forward_ms_median": round(statistics.median(fwd), 2),
            "total_ms_median": round(statistics.median(prep) + statistics.median(fwd), 2),
            "timed_images": len(fwd)}


def check_same_splits(run_dir: Path, split_by_seed: dict, paths: list[str]) -> None:
    """Stop if these splits differ from the run we are being compared with."""
    file = run_dir / "splits.csv"
    if not file.is_file():
        raise SystemExit(f"--compare-with: {file} not found")
    expected: dict[int, dict[str, str]] = {}
    with file.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            expected.setdefault(int(row["seed"]), {})[row["path"]] = row["split"]
    for seed, splits in split_by_seed.items():
        if seed not in expected:
            raise SystemExit(f"--compare-with: seed {seed} is not in {file}")
        if expected[seed] != dict(zip(paths, splits)):
            raise SystemExit(f"Splits for seed {seed} differ from {run_dir}. Use the same --train-dir, "
                             "ratios, --other-in-val/--other-in-train and seeds as that run.")
    print(f"Splits match {run_dir} for seeds {sorted(split_by_seed)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_args(parser)
    parser.add_argument("--compare-with", type=Path, default=None,
                        help="Run folder (e.g. outputs/embed_verify/stage1-pos) whose splits must match.")
    parser.add_argument("--laptop-labels", nargs="+", default=["laptop", "tv"],
                        help="Detector labels counted as 'laptop' (earlier benchmark: laptop + tv).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-latency", action="store_true")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.models = [MODEL_NAME]

    records, dedupe_report = load_records(args)
    out = make_out_dir(args, "baseline-rtdetrv2")
    unseen = frozenset(args.unseen_other)
    paths = [r.path for r in records]
    split_by_seed = {seed: split_for_seed(records, seed, args) for seed in args.seeds}
    if args.compare_with:
        check_same_splits(args.compare_with, split_by_seed, paths)
    write_run_config(out, args, "baseline", records, dedupe_report, {
        "split_counts": {str(s): count_table(records, sp)["split"] for s, sp in split_by_seed.items()},
        "detector": {"repo": REPO, "revision": REVISION, "laptop_labels": args.laptop_labels,
                     "detection_floor": DETECTION_FLOOR, "previous_threshold": PREVIOUS_THRESHOLD}})

    # Only val/test images of any seed are scored; train images are never needed.
    needed = sorted({i for sp in split_by_seed.values() for i, s in enumerate(sp) if s in ("val", "test")})
    print(f"Scoring {len(needed)} of {len(records)} images (val/test of all seeds), CPU time is "
          f"about 1 s per image...", flush=True)
    detector = Detector(args.device, args.laptop_labels)
    cache = args.out_root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    import hashlib
    key = hashlib.sha1(("3col-v2" + REVISION + ",".join(sorted(args.laptop_labels)) + "\n" +
                        "\n".join(records[i].sha256 for i in needed)).encode()).hexdigest()[:16]
    cache_file = cache / f"rtdetrv2-r50_scores_{key}.npz"
    S_all = np.full((len(records), 3), np.nan, dtype=np.float32)
    if cache_file.is_file():
        S_all[needed] = np.load(cache_file)["S"]
        print(f"scores loaded from {cache_file.name}", flush=True)
    else:
        from tqdm.auto import tqdm
        chunks = []
        for start in tqdm(range(0, len(needed), args.batch_size), desc="detect"):
            chunks.append(detector.scores([paths[i] for i in needed[start:start + args.batch_size]]))
        S_all[needed] = np.concatenate(chunks)
        np.savez(cache_file, S=S_all[needed])

    recorder = Recorder(unseen)
    for seed, splits in split_by_seed.items():
        keep = np.array([sp in ("val", "test") for sp in splits])
        S3 = np.full_like(S_all, np.nan)
        S3[keep] = S_all[keep]
        # (a) laptop/book scores only, as in the first baseline
        recorder.add(stage="baseline", model=MODEL_NAME, method=METHOD, seed=seed, S=S3[:, :2], records=records,
                     splits=splits, fixed_taus={"det005": PREVIOUS_THRESHOLD})
        # (b) the earlier detector rule: a laptop/book score only counts when it beats the strongest
        # other-object detection, so "det005" below reproduces the earlier benchmark exactly.
        S_full = S3[:, :2].copy()
        S_full[S3[:, :2] < S3[:, 2:3]] = 0.0
        recorder.add(stage="baseline", model=MODEL_NAME, method=METHOD_FULL, seed=seed, S=S_full, records=records,
                     splits=splits, fixed_taus={"det005": PREVIOUS_THRESHOLD})
    recorder.write(out)
    if not args.no_latency:
        row = latency(detector, paths)
        with (out / "latency.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    build_report([out])
    print(f"\nDone. Results: {out}")
    if args.compare_with:
        print(f"Combine with the embedding run:\n  python -m embed_verify.report --runs {args.compare_with} {out} "
              f"--out {out.parent / 'REPORT_all.md'}")


if __name__ == "__main__":
    main()
