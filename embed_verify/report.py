"""Build REPORT.md from one or more run folders.

    python -m embed_verify.report --runs outputs/embed_verify/stage1-... outputs/embed_verify/stage2-...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# RT-DETRv2-R50 Top-1 on the earlier 372-image detector benchmark (different data; reference only).
DETECTOR_REFERENCE = {"model": "RT-DETRv2-R50 (detector, Top-1)", "fnr": 0.0554, "fpr": 0.0099}


def pct(x) -> str:
    return "-" if pd.isna(x) else f"{100 * x:.1f}%"


def pm(mean, std) -> str:
    return "-" if pd.isna(mean) else f"{100 * mean:.1f} ± {100 * std:.1f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_no rows_\n"
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _load(runs: list[Path]):
    metrics, preds, latency, history = [], [], [], []
    for run in runs:
        for name, bucket in (("metrics.csv", metrics), ("predictions.csv", preds),
                             ("latency.csv", latency), ("history.csv", history)):
            file = run / name
            if file.is_file():
                df = pd.read_csv(file, encoding="utf-8-sig")
                df["run"] = run.name
                bucket.append(df)
    cat = lambda dfs: pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return cat(metrics), cat(preds), cat(latency), cat(history)


def _agg(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keys = ["model", "method", "split", "op_point"]
    g = df.groupby(keys)[cols]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=0).add_suffix("_std"))
    out["seeds"] = df.groupby(keys)["seed"].nunique()
    return out.reset_index()


def _main_table(agg: pd.DataFrame, split: str, op: str) -> str:
    sub = agg[(agg.split == split) & (agg.op_point == op)].copy()
    if sub.empty:
        return "_no rows_\n"
    sub["balanced_error"] = (sub.fnr_mean + sub.fpr_mean) / 2
    sub = sub.sort_values("balanced_error")
    rows = [[r.model, r.method, int(r.seeds), pm(r.accuracy_mean, r.accuracy_std),
             pm(r.macro_recall_mean, r.macro_recall_std), pm(r.fnr_mean, r.fnr_std),
             pm(r.swap_rate_mean, r.swap_rate_std), pm(r.fpr_mean, r.fpr_std),
             f"{r.auroc_mean:.3f}" if pd.notna(r.auroc_mean) else "-", pct(r.balanced_error)]
            for r in sub.itertuples()]
    return md_table(["model", "method", "seeds", "accuracy %", "macro recall %", "FNR %", "swap %", "FPR %",
                     "AUROC", "(FNR+FPR)/2"], rows)


def _per_class(agg: pd.DataFrame, split: str, op: str) -> str:
    sub = agg[(agg.split == split) & (agg.op_point == op)]
    rows = [[r.model, r.method, pm(r.recall_laptop_mean, r.recall_laptop_std), pm(r.recall_book_mean, r.recall_book_std),
             pm(r.recall_other_mean, r.recall_other_std), pct(r.precision_laptop_mean), pct(r.precision_book_mean)]
            for r in sub.itertuples()]
    return md_table(["model", "method", "laptop recall %", "book recall %", "other recall %",
                     "laptop precision", "book precision"], rows)


def _operating_points(agg: pd.DataFrame, split: str) -> str:
    sub = agg[(agg.split == split) & (agg.op_point != "oracle")]
    rows = [[r.model, r.method, r.op_point, pm(r.fnr_mean, r.fnr_std), pm(r.fpr_mean, r.fpr_std),
             pm(r.swap_rate_mean, r.swap_rate_std)] for r in sub.sort_values(["op_point", "fnr_mean"]).itertuples()]
    return md_table(["model", "method", "threshold rule", "test FNR %", "test FPR %", "swap %"], rows)


def _best_per_model(agg: pd.DataFrame) -> str:
    """Per model: the method with the lowest best-case error, next to its deployable thresholds."""
    test = agg[agg.split == "test"]
    oracle = test[test.op_point == "oracle"].copy()
    if oracle.empty:
        return "_no rows_\n"
    oracle["be"] = (oracle.fnr_mean + oracle.fpr_mean) / 2
    rows = []
    for model, group in oracle.groupby("model"):
        best = group.sort_values("be").iloc[0]
        row = [model, best.method, f"{best.tau_mean:.3f}", pm(best.fnr_mean, best.fnr_std),
               pm(best.fpr_mean, best.fpr_std), f"{best.auroc_mean:.3f}"]
        for op in ("q99", "q95"):
            dep = test[(test.model == model) & (test.method == best.method) & (test.op_point == op)]
            row += [pct(dep.fnr_mean.iloc[0]), pct(dep.fpr_mean.iloc[0])] if len(dep) else ["-", "-"]
        rows.append((best.be, row))
    return md_table(["model", "best method", "best-case tau", "FNR %", "FPR %", "AUROC",
                     "q99 FNR", "q99 FPR", "q95 FNR", "q95 FPR"], [r for _, r in sorted(rows, key=lambda x: x[0])])


def _subclass_fpr(metrics: pd.DataFrame, split: str, op: str, unseen: list[str]) -> str:
    sub = metrics[(metrics.split == split) & (metrics.op_point == op)]
    records = []
    for r in sub.itertuples():
        for name, value in json.loads(r.fpr_by_subclass).items():
            records.append({"model": r.model, "method": r.method, "subclass": name, "fpr": value})
    if not records:
        return "_no 'other' images in this split_\n"
    table = pd.DataFrame(records).groupby(["model", "method", "subclass"]).fpr.mean().unstack("subclass")
    headers = ["model", "method"] + [f"{c}{' (unseen)' if c in unseen else ''}" for c in table.columns]
    rows = [[m, me] + [pct(v) for v in vals] for (m, me), vals in table.iterrows()]
    return md_table(headers, rows)


def _errors(preds: pd.DataFrame, best: tuple, col: str, k: int = 10) -> str:
    model, method = best
    sub = preds[(preds.model == model) & (preds.method == method) & (preds.split == "test")].copy()
    if sub.empty:
        return ""
    sub["accept"] = sub[["score_laptop", "score_book"]].max(axis=1)
    fa = sub[(sub.true == "other") & (sub[col] != "other")].sort_values("accept", ascending=False).head(k)
    fr = sub[(sub.true != "other") & (sub[col] == "other")].sort_values("accept").head(k)
    sw = sub[(sub.true != "other") & (sub[col] != "other") & (sub.true != sub[col])]
    out = [f"Best configuration by (FNR+FPR)/2 on test: **{model} / {method}** (errors below pool all seeds).\n"]
    for title, df in (("Most confident false accepts (other accepted as laptop/book)", fa),
                      ("Least confident false rejects (laptop/book rejected as other)", fr)):
        out.append(f"**{title}**\n")
        out.append(md_table(["seed", "true", "subclass", "predicted", "accept score", "path"],
                            [[r.seed, r.true, r.subclass, getattr(r, col), f"{r.accept:.3f}", Path(r.path).name]
                             for r in df.itertuples()]))
    out.append(f"Laptop<->book swaps on test (all seeds): {len(sw)}\n")
    return "\n".join(out)


def _data_mode(args: dict) -> str:
    if args.get("other_in_train"):
        return "laptop/book + other (other also in validation and test)"
    if args.get("other_in_val"):
        return "laptop/book only; other split between validation (threshold) and test"
    return "laptop/book only; other used for testing only (thresholds come from laptop/book validation images)"


def build_report(runs: list[Path], out_path: Path | None = None) -> Path:
    runs = [Path(r) for r in runs]
    metrics, preds, latency, history = _load(runs)
    out_path = out_path or runs[0] / "REPORT.md"
    lines = ["# Embedding verification report\n"]
    if metrics.empty:
        out_path.write_text("# Embedding verification report\n\n_No metrics were produced._\n", encoding="utf-8")
        return out_path

    lines.append("## 1. Run setup\n")
    unseen: list[str] = []
    for run in runs:
        cfg_file = run / "run_config.json"
        if not cfg_file.is_file():
            continue
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        unseen = sorted(set(unseen) | set(cfg["args"].get("unseen_other") or []))
        data = cfg["data"]
        lines.append(f"**{run.name}** ({cfg['stage']}, {cfg['created']}, git {cfg['git_branch']} @ {cfg['git_commit'][:8]})\n")
        lines.append(f"- images by label: {data['label']}; targets by source: {data['source']}\n"
                     f"- 'other' sub-classes: {data['other_subclass']}; unseen-other (test only): {cfg['args'].get('unseen_other')}\n"
                     f"- training data: {_data_mode(cfg['args'])}\n"
                     f"- de-duplication: {cfg['dedupe']}\n"
                     f"- split counts per seed: {cfg.get('split_counts')}\n"
                     f"- models: {cfg['args'].get('models')}, seeds: {cfg['args'].get('seeds')}, device: {cfg['args'].get('device')}, "
                     f"torch {cfg['torch']}, transformers {cfg['transformers']}, torch threads {cfg['torch_threads']}\n")
    lines.append("Methods: `prototype` = cosine to class mean; `knn` = mean cosine to the 5 nearest training images "
                 "per class; `mahalanobis` = distance to class mean with a shared covariance; `logreg`/`mlp` = "
                 "classifier heads (score = softmax probability); `finetune-lastN` = fine-tuned backbone + cosine "
                 "classifier; `zeroshot-detector` = COCO-pretrained RT-DETRv2-R50, no training, score = highest laptop(+tv)/book "
                 "detection confidence; `zeroshot-detector-full` = the same detector with its original rule, where a laptop/book "
                 "score only counts if it beats the strongest other-object detection (`det005` = the 0.05 threshold of the earlier detector benchmark). A photo is "
                 "rejected as other when its best score is below the threshold.\n")
    lines.append("Definitions: **FNR** = laptop/book predicted as other; **swap** = laptop predicted as book or "
                 "vice versa; **FPR** = other predicted as laptop/book. Mean ± std over split seeds; results are on the "
                 "held-out test split.\n")
    lines.append("Threshold rules. **oracle** = the single best threshold per model, found on the test split itself "
                 "(minimises (FNR+FPR)/2). It removes threshold-selection noise when comparing models but is "
                 "optimistic and cannot be used in deployment. **q99 / q95** = accept 99% / 95% of the validation "
                 "laptop/book images (needs no `other` images). **macro, fpr5, fpr1** appear only when validation "
                 "contains `other`: max validation macro recall, or the loosest threshold with validation FPR <= 5% / 1%.\n")

    numeric = [c for c in ["accuracy", "macro_recall", "recall_laptop", "recall_book", "recall_other",
                           "precision_laptop", "precision_book", "fnr", "swap_rate", "fpr", "fpr_seen",
                           "fpr_unseen", "auroc", "ap", "tau"] if c in metrics]
    agg = _agg(metrics, numeric)
    agg.to_csv(out_path.parent / "summary_combined.csv", index=False, encoding="utf-8-sig")

    ops = set(metrics.op_point)
    primary = "macro" if "macro" in ops else "q99"
    lines.append("## 2. Model comparison at each model's best-case threshold (test, sorted by (FNR+FPR)/2)\n")
    lines.append(_main_table(agg, "test", "oracle"))
    lines.append("Best method per model, with the deployable thresholds of the same method:\n")
    lines.append(_best_per_model(agg))
    ref = DETECTOR_REFERENCE
    lines.append(f"Reference (different data, not comparable): {ref['model']} FNR {pct(ref['fnr'])}, FPR {pct(ref['fpr'])}.\n")
    lines.append("## 3. Deployable thresholds on test (chosen without looking at test)\n")
    lines.append(_operating_points(agg, "test"))
    lines.append(f"## 4. Per-class recall / precision on test (`{primary}`)\n")
    lines.append(_per_class(agg, "test", primary))
    lines.append(f"## 5. False-accept rate by 'other' sub-class on test (`{primary}`, mean over seeds)\n")
    lines.append(_subclass_fpr(metrics, "test", primary, unseen))
    if "fpr_unseen" in agg and agg[(agg.split == "test") & (agg.op_point == primary)].fpr_unseen_mean.notna().any():
        lines.append(f"## 6. Open-set check: FPR on unseen vs seen 'other' sub-classes (test, `{primary}`)\n")
        sub = agg[(agg.split == "test") & (agg.op_point == primary)]
        lines.append(md_table(["model", "method", "FPR seen %", "FPR unseen %"],
                              [[r.model, r.method, pm(r.fpr_seen_mean, r.fpr_seen_std), pm(r.fpr_unseen_mean, r.fpr_unseen_std)]
                               for r in sub.itertuples()]))
    if not latency.empty:
        lines.append("## 7. Speed and size (batch size 1, per image)\n")
        lines.append(md_table(["model", "params (M)", "embedding dim", "device", "threads", "preprocess ms", "forward ms", "total ms"],
                              [[r.model, r.params_m, r.embedding_dim, r.device, r.threads, r.preprocess_ms_median,
                                r.forward_ms_median, r.total_ms_median] for r in latency.drop_duplicates("model").itertuples()]))
    if not history.empty:
        lines.append("## 8. Fine-tuning history (validation balanced accuracy per epoch)\n")
        pivot = history.pivot_table(index=["model", "seed"], columns="epoch", values="val_balanced_acc")
        lines.append(md_table(["model", "seed"] + [f"ep{c}" for c in pivot.columns],
                              [[m, s] + [f"{v:.3f}" for v in vals] for (m, s), vals in pivot.iterrows()]))
        ft = metrics[metrics.method.str.startswith("finetune") & (metrics.split == "test") & (metrics.op_point == primary)]
        if "trainable_params_m" in ft:
            lines.append(md_table(["model", "seed", "trainable params (M)", "best val bal. acc", "total train s"],
                                  [[r.model, r.seed, r.trainable_params_m, r.best_val_balanced_acc,
                                    round(history[(history.model == r.model) & (history.seed == r.seed)].seconds.sum())]
                                   for r in ft.itertuples()]))
    test_best = agg[(agg.split == "test") & (agg.op_point == ("oracle" if "oracle" in ops else primary))]
    if not test_best.empty:
        best_row = test_best.assign(be=(test_best.fnr_mean + test_best.fpr_mean) / 2).sort_values("be").iloc[0]
        lines.append(f"## 9. Error analysis (threshold rule `{primary}`)\n")
        lines.append(_errors(preds, (best_row.model, best_row.method), f"pred_{primary}"))
    lines.append("## 10. Files\n\n`metrics.csv` (every model/method/seed/split/operating point, incl. confusion matrices), "
                 "`summary.csv`, `predictions.csv` (per image), `splits.csv`, `data_manifest.csv`, `run_config.json`"
                 ", `latency.csv`, `history.csv` (stage 2).\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    print(build_report(args.runs, args.out))


if __name__ == "__main__":
    main()
