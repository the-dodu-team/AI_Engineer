"""Image discovery, de-duplication and deterministic stratified splits."""
from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

CLASSES = ("laptop", "book", "other")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ALIASES = {"laptop": "laptop", "laptops": "laptop", "computer": "laptop", "computers": "laptop",
           "book": "book", "books": "book"}


@dataclass
class Record:
    path: str
    label: str      # laptop | book | other
    subclass: str   # folder name for "other"; equals label for targets
    source: str     # dataset source (from manifest.csv) or "unknown"
    sha256: str
    origin: str     # train | other

    @property
    def y(self) -> int:
        return CLASSES.index(self.label)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_sources(root: Path) -> dict[str, str]:
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        return {}
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        return {row.get("filename", ""): row.get("source", "unknown") for row in csv.DictReader(handle)}


def discover(root: Path, origin: str, force_other: bool = False,
             skip_target_names: bool = False) -> tuple[list[Record], int]:
    """Read images below `root`. The first folder name decides the label.

    With `force_other`, every image is "other" and the first folder name is its subclass.
    `skip_target_names` ignores folders named laptop/book (and aliases) so they can never become "other".
    Returns the records and the number of unreadable images that were skipped.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data folder not found: {root}")
    sources = _manifest_sources(root)
    records, bad = [], 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        parts = path.relative_to(root).parts
        first = parts[0].strip().lower().replace(" ", "_") if len(parts) > 1 else "root"
        if skip_target_names and first in ALIASES:
            continue
        label = "other" if force_other else ALIASES.get(first, "other")
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            bad += 1
            continue
        records.append(Record(
            path=str(path), label=label, subclass=first if label == "other" else label,
            source=sources.get(path.name, "unknown"), sha256=sha256_file(path), origin=origin))
    return records, bad


def dedupe(groups: list[list[Record]]) -> tuple[list[Record], dict]:
    """Merge record lists in priority order, dropping repeated SHA-256 hashes.

    An exact duplicate must not land in two splits, so each image is kept only once.
    """
    seen: dict[str, Record] = {}
    kept, report = [], {"duplicates_dropped": 0, "label_conflicts": 0}
    for group in groups:
        for rec in group:
            first = seen.get(rec.sha256)
            if first is None:
                seen[rec.sha256] = rec
                kept.append(rec)
                continue
            report["duplicates_dropped"] += 1
            if first.label != rec.label:
                report["label_conflicts"] += 1
    return kept, report


def limit_per_group(records: list[Record], n: int | None, seed: int = 0) -> list[Record]:
    """Keep at most `n` images per (label, subclass, source) group. Used for smoke runs."""
    if not n:
        return records
    rng = random.Random(seed)
    groups: dict[tuple, list[Record]] = defaultdict(list)
    for rec in records:
        groups[(rec.label, rec.subclass, rec.source, rec.origin)].append(rec)
    kept = []
    for key in sorted(groups):
        items = sorted(groups[key], key=lambda r: r.path)
        rng.shuffle(items)
        kept.extend(items[:n])
    return sorted(kept, key=lambda r: r.path)


def make_splits(records: list[Record], seed: int, unseen_other: set[str] | frozenset = frozenset(),
                ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
                other_in_train: bool = False, other_in_val: bool = False) -> list[str]:
    """Return a split name (train/val/test) per record.

    Stratified per (label, source) for targets and per subclass for "other", so every
    split holds both sources. `ratios` are (train, val, test).
    By default "other" images are used for testing only: models and thresholds come from
    laptop/book alone. `other_in_val` shares "other" between val (to tune the threshold)
    and test; `other_in_train` additionally lets it into train.
    Subclasses in `unseen_other` always go entirely to test.
    """
    splits = ["train"] * len(records)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        if rec.label == "other" and rec.subclass in unseen_other:
            splits[i] = "test"
        elif rec.label == "other":
            groups[("other", rec.subclass)].append(i)
        else:
            groups[(rec.label, rec.source)].append(i)
    rng = random.Random(seed)
    for key in sorted(groups):
        idx = sorted(groups[key], key=lambda i: records[i].path)
        rng.shuffle(idx)
        n = len(idx)
        if key[0] == "other" and not other_in_train and not other_in_val:
            n_val, n_test = 0, n  # default: "other" is only ever used for testing
        elif key[0] == "other" and not other_in_train:
            n_test = round(n * ratios[2] / (ratios[1] + ratios[2]))
            if n >= 2:
                n_test = min(max(n_test, 1), n - 1)
            n_val = n - n_test
        else:
            n_val = round(n * ratios[1])
            n_test = round(n * ratios[2])
            if n >= 5:
                n_val, n_test = max(n_val, 1), max(n_test, 1)
        for i in idx[:n_test]:
            splits[i] = "test"
        for i in idx[n_test:n_test + n_val]:
            splits[i] = "val"
    return splits


def count_table(records: list[Record], splits: list[str] | None = None) -> dict:
    """Counts by label, subclass and (optionally) split, for the run report."""
    table = {"label": dict(Counter(r.label for r in records)),
             "source": dict(Counter(f"{r.label}/{r.source}" for r in records if r.label != "other")),
             "other_subclass": dict(Counter(r.subclass for r in records if r.label == "other"))}
    if splits is not None:
        table["split"] = {s: dict(Counter(r.label for r, sp in zip(records, splits) if sp == s))
                          for s in ("train", "val", "test")}
    return table
