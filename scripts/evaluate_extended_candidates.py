"""Evaluate five additional pretrained detectors against the saved RTMDet manifest.

Each model runs in its own process so YOLOv13's Ultralytics fork cannot replace
the installed Ultralytics package. No training or threshold fitting is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
import zipfile

from PIL import Image

from evaluate_coco_candidates import GROUPS, decide, discover, metrics, write_csv

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = 1
YOLO13_REVISION = "73289949533efac82bb5f72ec19b746618656bd2"
CANDIDATES = {
    "rf-detr-large": {
        "repo": "Roboflow/rf-detr-large", "revision": "f62f7dd5252b61097cbace33886045816dadbde9",
        "params_m": 33.9, "coco_ap": 56.5, "code_license": "Apache-2.0",
        "commercial_status": "permissive_with_notices", "resolution": 704,
        "license_url": "https://github.com/roboflow/rf-detr/blob/develop/LICENSE",
        "license_note": "Official model card and RF-DETR table designate this checkpoint Apache-2.0. Preserve license/attribution and applicable NOTICE; no source-disclosure condition.",
    },
    "dfine-l-obj2coco": {
        "repo": "ustc-community/dfine-large-obj2coco-e25", "revision": "c89de47153e338c07cddc81fc21a776b97cbf75e",
        "params_m": 31.0, "coco_ap": 57.3, "code_license": "Apache-2.0",
        "commercial_status": "objects365_terms_review", "resolution": 640,
        "license_url": "https://github.com/Peterande/D-FINE#model-zoo",
        "license_note": "Code/model card Apache-2.0; upstream explicitly says Objects365 checkpoints must not be assumed commercially cleared. Obtain clearance before commercial deployment.",
    },
    "lw-detr-large": {
        "repo": "AnnaZhang/lwdetr_large_60e_coco", "revision": "1a96e2659a637027892e987104556b8a7ec7aac9",
        "params_m": 46.8, "coco_ap": 56.1, "code_license": "Apache-2.0",
        "commercial_status": "objects365_terms_review", "resolution": 640,
        "license_url": "https://github.com/Atten4Vis/LW-DETR",
        "license_note": "Code/converted model card Apache-2.0; original recipe uses Objects365 pretraining. Dataset website says academic purposes only; applicability to commercial weights needs clarification.",
    },
    "rf-detr-medium": {
        "repo": "Roboflow/rf-detr-medium", "revision": "1b5b672408f86dd38e05dd3cf3f2e0834e545a59",
        "params_m": 33.7, "coco_ap": 54.7, "code_license": "Apache-2.0",
        "commercial_status": "permissive_with_notices", "resolution": 576,
        "license_url": "https://github.com/roboflow/rf-detr/blob/develop/LICENSE",
        "license_note": "Official model card and RF-DETR table designate this checkpoint Apache-2.0. Preserve license/attribution and applicable NOTICE; no source-disclosure condition.",
    },
    "yolov13l": {
        "repo": "https://github.com/iMoonLab/yolov13", "revision": YOLO13_REVISION,
        "weights_url": "https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13l.pt",
        "params_m": 27.6, "coco_ap": 53.4, "code_license": "AGPL-3.0",
        "commercial_status": "copyleft_or_separate_permission", "resolution": 640,
        "license_url": "https://github.com/iMoonLab/yolov13/blob/main/LICENSE",
        "license_note": "Commercial use is not prohibited, but distribution/modified network-service obligations apply. A closed-source product needs an applicable separate grant from the relevant rights holders; do not assume an Ultralytics license covers this fork.",
    },
}
CORRECTION_CANDIDATES = {}
for _name, _repo, _revision, _params, _ap in (
    ("rtdetrv2-r50", "PekingU/rtdetr_v2_r50vd", "282494075698cab9faa1096ae26856890030c817", 43.0, 53.4),
    ("rtdetrv2-r34", "PekingU/rtdetr_v2_r34vd", "5d60d8fb5eb2a3ee5575bf200ecbce4e2cfa420c", 31.5, 49.9),
    ("rtdetr-r50", "PekingU/rtdetr_r50vd", "df939e661d8c52e80608d1ec566561aabd25a4e7", 43.0, 53.1),
    ("rtdetr-r18", "PekingU/rtdetr_r18vd", "ac77a11ff0170a41b771c03264987f8ce2b0d753", 20.2, 46.5),
):
    CORRECTION_CANDIDATES[_name] = {
        "repo": _repo, "revision": _revision, "params_m": _params, "coco_ap": _ap,
        "resolution": 640, "code_license": "Apache-2.0", "commercial_status": "permissive_with_notices",
        "license_url": "https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE",
        "license_note": "Apache-2.0 code and model card. Re-evaluated with tvmonitor correctly mapped to computer.",
    }
ALL_CANDIDATES = {**CANDIDATES, **CORRECTION_CANDIDATES}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def canonical_label(label: str) -> str:
    # D-FINE uses the old COCO name tvmonitor; the task's group is unchanged.
    label = label.strip().lower()
    return "tv" if label in {"tvmonitor", "television", "tv monitor"} else label


def validate_manifest(records: list[dict], baseline: dict) -> None:
    expected = {(row["sha256"], row["source_label"] if row["source_label"] in GROUPS else "other")
                for row in baseline["images"]}
    actual = {(row["sha256"], row["true_label"]) for row in records}
    if len(actual) != len(records) or len(expected) != len(baseline["images"]):
        raise ValueError("Duplicate hashes or labels in an evaluation manifest")
    if actual != expected:
        raise ValueError(f"Dataset differs from RTMDet manifest: missing={len(expected-actual)}, extra={len(actual-expected)}")
    if baseline.get("smoke_test"):
        raise ValueError("RTMDet baseline must be a full run")


def download(url: str, path: Path) -> None:
    if path.is_file():
        return
    partial = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    partial.replace(path)


def prepare_yolo13(cache: Path) -> Path:
    archive = cache / "source.zip"
    source = cache / f"yolov13-{YOLO13_REVISION}"
    if not source.is_dir():
        download(f"https://codeload.github.com/iMoonLab/yolov13/zip/{YOLO13_REVISION}", archive)
        with zipfile.ZipFile(archive) as zipped:
            for info in zipped.infolist():
                if not (cache / info.filename).resolve().is_relative_to(cache.resolve()):
                    raise ValueError("Unsafe source archive path")
            zipped.extractall(cache)
    download(CANDIDATES["yolov13l"]["weights_url"], cache / "yolov13l.pt")
    return source


class Detector:
    def __init__(self, name: str, cache: Path, device: str):
        import torch

        self.name, self.device = name, device
        spec = ALL_CANDIDATES[name]
        cache.mkdir(parents=True, exist_ok=True)
        if name == "yolov13l":
            source = prepare_yolo13(cache)
            sys.path.insert(0, str(source))
            from ultralytics import YOLO
            import ultralytics
            from ultralytics.utils import torch_utils

            if not Path(ultralytics.__file__).resolve().is_relative_to(source.resolve()):
                raise RuntimeError("YOLOv13 requires its pinned source in a fresh worker")
            # select_device() otherwise resets CPU inference to the fork's 8-thread default.
            torch_utils.NUM_THREADS = torch.get_num_threads()
            checkpoint = cache / "yolov13l.pt"
            self.model = YOLO(str(checkpoint))
            labels = self.model.names.values()
            module = self.model.model
            self.artifacts = [{"file": checkpoint.name, "sha256": sha256(checkpoint)}]
            self.preprocess = {"imgsz": 640, "letterbox": True, "nms_iou": 0.6, "max_det": 300}
        else:
            from huggingface_hub import snapshot_download
            from transformers import AutoImageProcessor, AutoModelForObjectDetection

            snapshot = Path(snapshot_download(
                spec["repo"], revision=spec["revision"], cache_dir=str(cache / "huggingface"),
                allow_patterns=["*.json", "*.safetensors", "README.md", "LICENSE*"],
            ))
            self.processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
            self.model = AutoModelForObjectDetection.from_pretrained(snapshot, local_files_only=True).to(device).eval()
            labels = self.model.config.id2label.values()
            module = self.model
            self.artifacts = [{"file": p.name, "sha256": sha256(p)} for p in sorted(snapshot.glob("*.safetensors"))]
            self.preprocess = self.processor.to_dict()
        label_names = {canonical_label(label) for label in labels}
        if not {"book", "laptop", "tv"}.issubset(label_names):
            raise ValueError(f"Unexpected class vocabulary: {label_names}")
        self.parameter_count = sum(p.numel() for p in module.parameters())
        self.torch_version = torch.__version__

    def predict(self, image: Image.Image) -> list[tuple[str, float]]:
        import torch

        if self.name == "yolov13l":
            result = self.model.predict(image, imgsz=640, conf=0.01, iou=0.6,
                                        max_det=300, device=self.device, verbose=False)[0]
            return [(canonical_label(result.names[int(label)]), float(score))
                    for label, score in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())]
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_object_detection(
            outputs, target_sizes=[(image.height, image.width)], threshold=0.01,
        )[0]
        return [(canonical_label(self.model.config.id2label[int(label)]), float(score))
                for label, score in zip(result["labels"].tolist(), result["scores"].tolist())]


def sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)


def run_worker(args) -> None:
    import torch

    torch.set_num_threads(args.threads)
    name = args.worker
    manifest = json.loads((args.output / "run.json").read_text(encoding="utf-8"))
    records = manifest["images"]
    cache = (args.cache_dir / name).resolve()
    output = args.output / name
    output.mkdir(parents=True, exist_ok=True)
    partial_path = output / "predictions.partial.csv"
    detector = Detector(name, cache, args.device)
    print(f"Loaded {name}: {detector.parameter_count/1e6:.3f}M parameters", flush=True)
    with Image.open(records[0]["path"]) as first:
        warmup = first.convert("RGB")
    for _ in range(args.warmup):
        detector.predict(warmup)
    if torch.get_num_threads() != args.threads:
        raise RuntimeError("Backend changed the requested CPU thread count")
    rows = []
    if partial_path.is_file():
        with partial_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected_prefix = [(row["sha256"], row["true_label"]) for row in records[:len(rows)]]
        actual_prefix = [(row["sha256"], row["true_label"]) for row in rows]
        if actual_prefix != expected_prefix:
            raise ValueError("Partial predictions do not match the current manifest prefix")
        print(f"Resuming {name} at {len(rows)}/{len(records)}", flush=True)
    for index, record in enumerate(records[len(rows):], len(rows) + 1):
        with Image.open(record["path"]) as source:
            image = source.convert("RGB")
        timings = []
        for _ in range(args.repeats):
            sync(args.device)
            start = time.perf_counter()
            detections = detector.predict(image)
            sync(args.device)
            timings.append((time.perf_counter() - start) * 1000)
        prediction, scores = decide(detections, args.threshold)
        raw = max(detections, key=lambda item: item[1], default=("no_detection", 0.0))
        rows.append({**record, "prediction": prediction,
                     **{f"score_{group}": scores[group] for group in GROUPS},
                     "raw_top1": raw[0], "raw_top1_score": raw[1],
                     "latency_ms": statistics.mean(timings)})
        if index % 25 == 0 or index == len(records):
            write_csv(partial_path, rows)
            print(f"{name}: {index}/{len(records)}", flush=True)
    write_csv(output / "predictions.csv", rows)
    result = metrics(rows)
    result.update({"model": name, **ALL_CANDIDATES[name], "actual_params_m": detector.parameter_count/1e6,
                   "artifacts": detector.artifacts, "preprocessing": detector.preprocess,
                   "protocol": manifest["protocol"], "manifest_sha256": manifest["manifest_sha256"],
                   "torch": detector.torch_version,
                   "transformers": importlib.metadata.version("transformers"),
                   "predictions_sha256": sha256(output / "predictions.csv")})
    write_json(output / "metrics.json", result)
    partial_path.unlink(missing_ok=True)
    print(f"{name}: accuracy={result['accuracy']:.2%}, FPR={result['fpr']:.2%}, FNR={result['fnr']:.2%}", flush=True)
    if not args.keep_weights:
        # Only remove this worker's known cache beneath the configured cache root.
        if cache == args.cache_dir.resolve() or not cache.is_relative_to(args.cache_dir.resolve()):
            raise ValueError("Cache cleanup escaped its root")
        del detector
        import gc
        gc.collect()
        shutil.rmtree(cache)


def completed_result(output: Path, name: str, manifest: dict) -> dict | None:
    metrics_file, predictions_file = output / name / "metrics.json", output / name / "predictions.csv"
    if not metrics_file.is_file() or not predictions_file.is_file():
        return None
    result = json.loads(metrics_file.read_text(encoding="utf-8"))
    if (result.get("protocol") != manifest["protocol"] or
            result.get("manifest_sha256") != manifest["manifest_sha256"] or
            result.get("revision") != ALL_CANDIDATES[name]["revision"] or
            result.get("predictions_sha256") != sha256(predictions_file)):
        raise ValueError(f"Saved {name} result has a different protocol, manifest, or artifact")
    with predictions_file.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [(row["sha256"], row["true_label"]) for row in rows] != [(row["sha256"], row["true_label"]) for row in manifest["images"]]:
        raise ValueError(f"Incomplete or reordered predictions for {name}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/coco_extended_detectors")
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/rtmdet/model-size-20260927/run.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "models/temporary/extended-candidates")
    parser.add_argument("--models", nargs="+", choices=list(ALL_CANDIDATES), default=list(CANDIDATES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-weights", action="store_true")
    parser.add_argument("--worker", choices=list(ALL_CANDIDATES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0.01 <= args.threshold <= 1 or args.warmup < 0 or args.repeats < 1 or args.threads < 1:
        parser.error("Invalid threshold, warmup, repeats, or thread count")
    for key in ("source", "output", "baseline", "cache_dir"):
        setattr(args, key, getattr(args, key).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(args.cache_dir / "ultralytics-settings")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_XET_CACHE"] = str(args.cache_dir / "xet")
    os.environ["PYTHONUTF8"] = "1"
    if args.worker:
        run_worker(args)
        return
    images = discover(args.source)
    records = [{"path": str(path), "true_label": label,
                "source_label": path.relative_to(args.source).parts[0].lower(), "sha256": sha256(path)}
               for path, label in images]
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    validate_manifest(records, baseline)
    fingerprint = hashlib.sha256(json.dumps([(r["sha256"], r["true_label"]) for r in records]).encode()).hexdigest()
    manifest = {"images": records, "manifest_sha256": fingerprint,
                "source": str(args.source), "baseline": str(args.baseline), "models": args.models,
                "protocol": {"version": PROTOCOL_VERSION, "threshold": args.threshold, "detection_floor": 0.01,
                             "warmup": args.warmup, "repeats": args.repeats, "threads": args.threads,
                             "device": args.device, "input": "checkpoint-native", "top_k": 1},
                "python": sys.version, "platform": platform.platform(),
                "timing_note": "Decoded image preprocessing + inference + postprocessing; no disk I/O/model loading. Historical runs used other runtimes/repeats; latency is descriptive, not a controlled speed comparison."}
    run_path = args.output / "run.json"
    if run_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        if not args.resume or previous["manifest_sha256"] != fingerprint or previous["protocol"] != manifest["protocol"]:
            parser.error("Output already exists or differs; use a new output path, or --resume for the identical run")
    write_json(run_path, manifest)
    print(f"Verified {len(records)} unique images against RTMDet SHA-256 manifest", flush=True)
    results, failures = [], {}
    for name in args.models:
        result = completed_result(args.output, name, manifest) if args.resume else None
        if result is None:
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", name,
                       "--source", str(args.source), "--output", str(args.output), "--cache-dir", str(args.cache_dir),
                       "--device", args.device, "--threshold", str(args.threshold), "--warmup", str(args.warmup),
                       "--repeats", str(args.repeats), "--threads", str(args.threads)]
            if args.keep_weights:
                command.append("--keep-weights")
            model_output = args.output / name
            model_output.mkdir(exist_ok=True)
            with (model_output / "inference.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace", cwd=ROOT)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    # Windows consoles can use CP949, while model logs contain emoji.
                    console = getattr(sys.stdout, "encoding", None) or "utf-8"
                    print(line.encode(console, errors="replace").decode(console), end="", flush=True)
                code = process.wait()
            if code:
                failures[name] = f"worker exited {code}; see {name}/inference.log"
                print(f"FAILED: {name}", flush=True)
                continue
            result = completed_result(args.output, name, manifest)
        results.append(result)
        write_csv(args.output / "summary.csv", [{key: row[key] for key in (
            "model", "images", "accuracy", "fpr", "fnr", "false_accepts", "false_rejects",
            "mean_latency_ms", "actual_params_m", "params_m", "coco_ap", "code_license", "commercial_status",
        )} for row in results])
    write_json(args.output / "status.json", {"completed": [r["model"] for r in results], "failures": failures})
    if failures:
        raise SystemExit(f"Incomplete models: {', '.join(failures)}")


if __name__ == "__main__":
    main()
