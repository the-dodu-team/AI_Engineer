"""Stage 2: fine-tune the last N backbone blocks plus a cosine-classifier head.

Uses the same seeds and splits as stage 1, so results are directly comparable.
Run it for the models that did best in stage 1:

    python -m embed_verify.stage2_finetune --train-dir <...> --other-dir <...> --models dinov2-small
"""
from __future__ import annotations

import argparse
import copy
import csv
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .backbones import Backbone, extract, load_image
from .data import count_table
from .evalcore import Recorder
from .report import build_report
from .runinfo import add_data_args, load_records, make_out_dir, split_for_seed, write_run_config


class CosineHead(nn.Module):
    """logits = scale * cos(embedding, class weight): an embedding-style classifier."""

    def __init__(self, dim: int, n_cls: int, scale: float = 20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_cls, dim) * 0.02)
        self.scale = scale

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.scale * nn.functional.normalize(emb, dim=1) @ nn.functional.normalize(self.weight, dim=1).T


class TrainSet(Dataset):
    def __init__(self, paths, labels, size, mean, std):
        self.paths, self.labels = paths, labels
        self.tf = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.5, 1.0)), transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(load_image(self.paths[i])), self.labels[i]


def embed_all(backbone, paths, batch_size):
    return torch.tensor(extract(backbone, paths, batch_size, desc="eval"))


def bal_acc(logits: np.ndarray, y: np.ndarray) -> float:
    pred = logits.argmax(1)
    recalls = [(pred[y == c] == c).mean() for c in np.unique(y)]
    return float(np.mean(recalls)) if recalls else 0.0


def finetune_one(args, model_name, seed, records, splits):
    torch.manual_seed(seed)
    np.random.seed(seed)
    backbone = Backbone(model_name, args.device, args.dinov2_pool)
    y = np.array([r.y for r in records])
    split_arr = np.array(splits)
    idx = {s: np.flatnonzero(split_arr == s) for s in ("train", "val", "test")}
    n_cls = 3 if (y[idx["train"]] == 2).any() else 2
    paths = [r.path for r in records]

    trainable = backbone.last_blocks(args.unfreeze_blocks) + backbone.final_modules()
    for module in trainable:
        for p in module.parameters():
            p.requires_grad_(True)
    dim = int(backbone.embed(backbone.preprocess([load_image(paths[0])])).shape[1])
    head = CosineHead(dim, n_cls).to(args.device)
    backbone_params = [p for m in trainable for p in m.parameters()]
    opt = torch.optim.AdamW([{"params": backbone_params, "lr": args.lr_backbone},
                             {"params": head.parameters(), "lr": args.lr_head}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    counts = np.bincount(y[idx["train"]], minlength=n_cls)[:n_cls].astype(float)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / np.clip(counts, 1, None) / n_cls,
                                                      dtype=torch.float32).to(args.device), label_smoothing=0.1)
    mean, std = backbone.mean_std
    loader = DataLoader(TrainSet([paths[i] for i in idx["train"]], y[idx["train"]].tolist(),
                                 backbone.image_size, mean, std),
                        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)

    def snapshot():
        return copy.deepcopy([m.state_dict() for m in trainable]), copy.deepcopy(head.state_dict())

    best, best_score, history = snapshot(), -1.0, []
    val_mask = y[idx["val"]] < n_cls  # a 2-class head cannot score "other" validation images
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        for m in trainable:
            m.train()
        losses = []
        for pixels, labels in loader:
            opt.zero_grad()
            loss = loss_fn(head(backbone.embed(pixels)), labels.to(args.device))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        for m in trainable:
            m.eval()
        with torch.no_grad():
            val_emb = embed_all(backbone, [paths[i] for i in idx["val"]], args.batch_size)
            val_logits = head(val_emb.to(args.device)).cpu().numpy()
        score = bal_acc(val_logits[val_mask], y[idx["val"]][val_mask]) if val_mask.any() else 0.0
        history.append({"model": model_name, "seed": seed, "epoch": epoch,
                        "train_loss": round(float(np.mean(losses)), 4), "val_balanced_acc": round(score, 4),
                        "seconds": round(time.perf_counter() - t0, 1)})
        print(f"  seed {seed} epoch {epoch}/{args.epochs} loss {history[-1]['train_loss']} "
              f"val_bal_acc {score:.3f} ({history[-1]['seconds']}s)", flush=True)
        if score >= best_score:
            best_score, best = score, snapshot()
    for module, state in zip(trainable, best[0]):
        module.load_state_dict(state)
    head.load_state_dict(best[1])

    S = np.full((len(records), 2), np.nan, dtype=np.float32)
    with torch.no_grad():
        for split in ("val", "test"):
            if not len(idx[split]):
                continue
            emb = embed_all(backbone, [paths[i] for i in idx[split]], args.batch_size)
            S[idx[split]] = torch.softmax(head(emb.to(args.device)), 1)[:, :2].cpu().numpy()
    extra = {"best_val_balanced_acc": round(best_score, 4), "unfreeze_blocks": args.unfreeze_blocks,
             "epochs": args.epochs, "trainable_params_m": round(
                 (sum(p.numel() for p in backbone_params) + sum(p.numel() for p in head.parameters())) / 1e6, 2)}
    return S, history, extra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_args(parser)
    parser.set_defaults(models=["dinov2-small"])
    parser.add_argument("--unfreeze-blocks", type=int, default=2, help="Number of last transformer blocks to train.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--dinov2-pool", default="cls_mean", choices=["cls", "cls_mean"])
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    records, dedupe_report = load_records(args)
    out = make_out_dir(args, "stage2")
    unseen = frozenset(args.unseen_other)
    split_by_seed = {seed: split_for_seed(records, seed, args) for seed in args.seeds}
    write_run_config(out, args, "stage2", records, dedupe_report, {
        "split_counts": {str(s): count_table(records, sp)["split"] for s, sp in split_by_seed.items()}})
    print(f"{len(records)} images -> {out}", flush=True)

    recorder, histories = Recorder(unseen), []
    method = f"finetune-last{args.unfreeze_blocks}"
    for model_name in args.models:
        for seed in args.seeds:
            print(f"\n== {model_name} seed {seed}", flush=True)
            S, history, extra = finetune_one(args, model_name, seed, records, split_by_seed[seed])
            histories += history
            recorder.add(stage="stage2", model=model_name, method=method, seed=seed, S=S,
                         records=records, splits=split_by_seed[seed], extra=extra)
            recorder.write(out)
            with (out / "history.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(histories[0]))
                writer.writeheader()
                writer.writerows(histories)
    build_report([out])
    print(f"\nDone. Results: {out}\nGive {out / 'REPORT.md'} to Claude for the write-up.")


if __name__ == "__main__":
    main()
