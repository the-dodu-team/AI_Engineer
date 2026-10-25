"""Create a tiny synthetic image set to check that the pipeline runs end to end.

    python -m embed_verify.make_synthetic outputs/embed_verify/_synthetic
Creates <dir>/train/{laptop,book} and <dir>/other/{bed,table,chair}.
The images are random shapes, so the metrics are meaningless; only the plumbing is tested.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw


def draw(kind: str, rng: random.Random, size: int = 160) -> Image.Image:
    image = Image.new("RGB", (size, size), tuple(rng.randint(0, 255) for _ in range(3)))
    d = ImageDraw.Draw(image)
    for _ in range(rng.randint(3, 8)):
        x, y = rng.randint(0, size - 40), rng.randint(0, size - 40)
        w, h = (rng.randint(60, 90), rng.randint(30, 45)) if kind == "laptop" else (
            (rng.randint(25, 40), rng.randint(60, 90)) if kind == "book" else (rng.randint(10, 60), rng.randint(10, 60)))
        d.rectangle([x, y, x + w, y + h], fill=tuple(rng.randint(0, 255) for _ in range(3)))
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--per-class", type=int, default=14)
    args = parser.parse_args()
    rng = random.Random(0)
    layout = {"train": ["laptop", "book"], "other": ["bed", "table", "chair"]}
    for part, classes in layout.items():
        for cls in classes:
            folder = args.out / part / cls
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(args.per_class):
                draw(cls if cls in {"laptop", "book"} else "other", rng).save(folder / f"{cls}_{i}.jpg")
    print(f"Wrote synthetic images under {args.out}")


if __name__ == "__main__":
    main()
