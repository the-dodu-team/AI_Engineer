"""Compare compact COCO detectors on the RTMDet MVP's image-level labels.

Downloads are isolated in a temporary directory and removed after each model.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
import statistics
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class Candidate:
    name: str
    backend: str
    checkpoint: str
    params_m: float
    coco_ap: float


# All ten checkpoints were trained for COCO detection and are below RTMDet-l's
# 52.3M parameters. The published AP is only a shortlist aid, not MVP accuracy.
CANDIDATES = (
    Candidate("yolo26l", "ultralytics", "yolo26l.pt", 24.8, 55.0),
    Candidate("yolo26m", "ultralytics", "yolo26m.pt", 20.4, 53.1),
    Candidate("yolo11l", "ultralytics", "yolo11l.pt", 25.3, 53.4),
    Candidate("yolo11m", "ultralytics", "yolo11m.pt", 20.1, 51.5),
    Candidate("yolov10l", "ultralytics", "yolov10l.pt", 24.4, 53.2),
    Candidate("yolov10m", "ultralytics", "yolov10m.pt", 15.4, 51.1),
    Candidate("rtdetrv2-r50", "transformers", "PekingU/rtdetr_v2_r50vd", 43.0, 53.4),
    Candidate("rtdetrv2-r34", "transformers", "PekingU/rtdetr_v2_r34vd", 31.5, 49.9),
    Candidate("rtdetr-r50", "transformers", "PekingU/rtdetr_r50vd", 43.0, 53.1),
    Candidate("rtdetr-r18", "transformers", "PekingU/rtdetr_r18vd", 20.2, 46.5),
)
BY_NAME = {item.name: item for item in CANDIDATES}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
GROUPS = ("computer", "book", "other")
EXPECTED = {"computer": 200, "book": 71, "other": 101}


def discover(source: Path) -> list[tuple[Path, str]]:
    if not source.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {source}")
    unique: dict[str, tuple[Path, str]] = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = path.relative_to(source).parts
        if len(parts) < 2:
            raise ValueError(f"Expected a class folder above image: {path}")
        folder = parts[0].lower()
        label = "computer" if folder in ("computer", "laptop") else "book" if folder == "book" else "other"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in unique:
            previous_path, previous_label = unique[digest]
            if previous_label != label:
                raise ValueError(f"Same image has conflicting labels: {previous_path} and {path}")
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Invalid image: {path}") from exc
        unique[digest] = (path, label)
    if not unique:
        raise ValueError(f"No images found in {source}")
    return list(unique.values())


def decide(detections: list[tuple[str, float]], threshold: float) -> tuple[str, dict[str, float]]:
    scores = dict.fromkeys(GROUPS, 0.0)
    for name, confidence in detections:
        name = name.lower().strip()
        # HF's COCO checkpoints use the legacy name tvmonitor for the same TV class.
        group = "computer" if name in ("laptop", "tv", "tvmonitor", "television", "tv monitor") else "book" if name == "book" else "other"
        scores[group] = max(scores[group], float(confidence))
    passing = [name for name in GROUPS if scores[name] >= threshold]
    return (max(passing, key=scores.__getitem__) if passing else "other"), scores


class Detector:
    def __init__(self, candidate: Candidate, device: str):
        self.candidate = candidate
        self.device = device
        if candidate.backend == "ultralytics":
            from ultralytics import YOLO

            self.model = YOLO(candidate.checkpoint)
        else:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection

            cache = str(Path.cwd() / "huggingface")
            self.processor = AutoImageProcessor.from_pretrained(candidate.checkpoint, cache_dir=cache)
            self.model = AutoModelForObjectDetection.from_pretrained(candidate.checkpoint, cache_dir=cache)
            self.model.to(device).eval()

    def predict(self, image: Image.Image) -> list[tuple[str, float]]:
        if self.candidate.backend == "ultralytics":
            result = self.model.predict(
                image, imgsz=640, conf=0.01, iou=0.6, max_det=300,
                device=self.device, verbose=False,
            )[0]
            if result.boxes is None:
                return []
            return [
                (str(result.names[int(class_id)]), float(score))
                for class_id, score in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())
            ]

        import torch

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model(**inputs)
        detections = self.processor.post_process_object_detection(
            output,
            target_sizes=torch.tensor([(image.height, image.width)]),
            threshold=0.01,
        )[0]
        return [
            (str(self.model.config.id2label[int(class_id)]), float(score))
            for class_id, score in zip(detections["labels"], detections["scores"])
        ]


def sync_device(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


def metrics(rows: list[dict]) -> dict:
    counts = Counter((row["true_label"], row["prediction"]) for row in rows)
    positives = sum(row["true_label"] != "other" for row in rows)
    negatives = len(rows) - positives
    false_rejects = sum(counts[(true, "other")] for true in ("computer", "book"))
    false_accepts = sum(counts[("other", prediction)] for prediction in ("computer", "book"))
    fpr = false_accepts / negatives
    fnr = false_rejects / positives
    return {
        "images": len(rows),
        "accuracy": sum(row["true_label"] == row["prediction"] for row in rows) / len(rows),
        "fpr": fpr,
        "fnr": fnr,
        "beats_rtmdet_x_fpr": fpr < 1 / 101,
        "beats_rtmdet_x_fnr": fnr < 21 / 271,
        "meets_fpr_5pct": fpr <= 0.05,
        "meets_fnr_5pct": fnr <= 0.05,
        "false_accepts": false_accepts,
        "negatives": negatives,
        "false_rejects": false_rejects,
        "positives": positives,
        "book_misses": counts[("book", "other")],
        "computer_misses": counts[("computer", "other")],
        "confusion": {true: {pred: counts[(true, pred)] for pred in GROUPS} for true in GROUPS},
        "mean_latency_ms": statistics.mean(row["latency_ms"] for row in rows),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def temporary_model_dir(parent: Path, name: str):
    # tempfile-created directories are inaccessible in some managed Windows setups.
    path = parent / f"{name}-{uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def evaluate(candidate: Candidate, images: list[tuple[Path, str]], args: argparse.Namespace) -> dict:
    model_dir = args.output / candidate.name
    model_dir.mkdir(parents=True, exist_ok=True)
    # Separate temporary caches keep peak disk use to one model at a time.
    with temporary_model_dir(args.cache_dir, candidate.name) as temporary:
        original_cwd = Path.cwd()
        detector = None
        try:
            os.chdir(temporary)
            detector = Detector(candidate, args.device)
            with Image.open(images[0][0]) as first:
                warmup_image = first.convert("RGB")
            for _ in range(args.warmup):
                detector.predict(warmup_image)
            rows = []
            for index, (path, truth) in enumerate(images, 1):
                with Image.open(path) as source_image:
                    image = source_image.convert("RGB")
                sync_device(args.device)
                start = time.perf_counter()
                detections = detector.predict(image)
                sync_device(args.device)
                elapsed_ms = (time.perf_counter() - start) * 1000
                prediction, scores = decide(detections, args.threshold)
                rows.append({
                    "path": str(path), "true_label": truth, "prediction": prediction,
                    "score_computer": scores["computer"], "score_book": scores["book"],
                    "score_other": scores["other"], "latency_ms": elapsed_ms,
                })
                if index % 25 == 0 or index == len(images):
                    print(f"{candidate.name}: {index}/{len(images)}", flush=True)
            write_csv(model_dir / "predictions.csv", rows)
            result = metrics(rows)
            result.update({"model": candidate.name, "backend": candidate.backend,
                           "checkpoint": candidate.checkpoint, "params_m": candidate.params_m,
                           "published_coco_ap": candidate.coco_ap})
            (model_dir / "metrics.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return result
        finally:
            os.chdir(original_cwd)
            del detector
            gc.collect()


def smoke(candidate: Candidate, image_path: Path, args: argparse.Namespace) -> None:
    with temporary_model_dir(args.cache_dir, candidate.name) as temporary:
        original_cwd = Path.cwd()
        detector = None
        try:
            os.chdir(temporary)
            detector = Detector(candidate, args.device)
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
            detections = detector.predict(image)
            prediction, scores = decide(detections, args.threshold)
            print(f"{candidate.name}: OK, {len(detections)} detections, "
                  f"prediction={prediction}, scores={scores}", flush=True)
        finally:
            os.chdir(original_cwd)
            del detector
            gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Root with computer/, book/, and other class folders")
    parser.add_argument("--output", type=Path, default=Path("outputs/coco_small_detectors"))
    parser.add_argument("--cache-dir", type=Path, default=Path("models/temporary"))
    parser.add_argument("--models", nargs="+", choices=list(BY_NAME), default=list(BY_NAME))
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--allow-different-sample", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Download and infer one book image per model")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()
    if args.list_models:
        for candidate in CANDIDATES:
            print(f"{candidate.name:14} {candidate.params_m:5.1f}M  COCO AP {candidate.coco_ap:4.1f}  {candidate.checkpoint}")
        return
    if args.source is None:
        parser.error("--source is required unless --list-models is used")
    if not 0.01 <= args.threshold <= 1 or args.warmup < 0:
        parser.error("--threshold must be 0.01..1 and --warmup must be nonnegative")
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.cache_dir = args.cache_dir.resolve()
    images = discover(args.source)
    counts = Counter(label for _, label in images)
    if dict(counts) != EXPECTED and not args.allow_different_sample:
        parser.error(
            f"Unique image counts {dict(counts)} differ from the RTMDet baseline {EXPECTED}; "
            "use --allow-different-sample only for a separate experiment"
        )
    if not all(counts[group] for group in GROUPS):
        parser.error("The dataset must include computer, book, and other images")
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    settings_dir = args.cache_dir / "ultralytics-settings"
    settings_dir.mkdir(exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(settings_dir)
    if args.smoke:
        book_image = next(path for path, label in images if label == "book")
        failures = []
        for name in args.models:
            try:
                smoke(BY_NAME[name], book_image, args)
            except Exception as exc:
                print(f"{name}: FAILED: {type(exc).__name__}: {exc}", flush=True)
                failures.append(name)
        if failures:
            raise SystemExit(f"Smoke check failed for: {', '.join(failures)}")
        return
    summary = []
    for name in args.models:
        result = evaluate(BY_NAME[name], images, args)
        summary.append({key: result[key] for key in (
            "model", "images", "accuracy", "fpr", "fnr", "false_accepts",
            "false_rejects", "book_misses", "computer_misses", "mean_latency_ms",
            "params_m", "published_coco_ap", "beats_rtmdet_x_fpr",
            "beats_rtmdet_x_fnr", "meets_fpr_5pct", "meets_fnr_5pct",
        )})
        write_csv(args.output / "summary.csv", summary)
        print(f"{name}: FPR={result['fpr']:.2%}, FNR={result['fnr']:.2%}, "
              f"ACC={result['accuracy']:.2%}, latency={result['mean_latency_ms']:.1f}ms")
    (args.output / "run.json").write_text(json.dumps({
        "source": str(args.source), "counts": dict(counts), "device": args.device,
        "threshold": args.threshold, "warmup": args.warmup,
        "models": args.models, "note": "Inference timing includes preprocessing and postprocessing, excluding image I/O and model loading.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
