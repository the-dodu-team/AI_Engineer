"""Threshold selection, metrics and result recording shared by stage 1 and stage 2.

Decision rule for every method: take the target class (laptop/book) with the highest
score S; accept it if S >= tau, otherwise predict "other". tau is always chosen on the
validation split, never on the test split.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import CLASSES, Record

OTHER = 2
# Operating points. "q99"/"q95" accept 99%/95% of the validation laptop/book images and need no
# "other" images; "macro"/"fpr5"/"fpr1" are tuned with validation "other" images when there are any.
OP_POINTS = {"q99": ("quantile", 0.01), "q95": ("quantile", 0.05),
             "macro": ("macro", None), "fpr5": ("fpr", 0.05), "fpr1": ("fpr", 0.01)}
OPS_NEEDING_OTHER = ("macro", "fpr5", "fpr1")
EVAL_SPLITS = ("val", "test")


def decide(S: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    cls, accept = S.argmax(1), S.max(1)
    return np.where(accept >= tau, cls, OTHER), accept


def _macro_recall(y: np.ndarray, pred: np.ndarray) -> float:
    recalls = [(pred[y == c] == c).mean() for c in range(3) if (y == c).any()]
    return float(np.mean(recalls)) if recalls else float("nan")


def _candidates(accept: np.ndarray) -> np.ndarray:
    u = np.unique(accept)
    mids = (u[:-1] + u[1:]) / 2 if len(u) > 1 else u
    return np.concatenate([[-np.inf], mids, [u.max() + 1e-6]])


def choose_tau(S: np.ndarray, y: np.ndarray, op: str) -> tuple[float, str]:
    """Pick the accept threshold on validation data. Returns (tau, note)."""
    cls, accept = S.argmax(1), S.max(1)
    has_other = (y == OTHER).any()
    has_target = (y < OTHER).any()
    kind, value = OP_POINTS[op]
    if not has_target:
        return 0.5, "no_target_in_val"
    if kind == "quantile":
        return float(np.quantile(accept[y < OTHER], value)), f"accept_{round(100 * (1 - value))}pct_of_val_targets"
    if not has_other:
        raise ValueError(f"Operating point {op!r} needs 'other' images in the validation split")
    cands = _candidates(accept)
    if kind == "macro":
        scores = np.array([_macro_recall(y, np.where(accept >= t, cls, OTHER)) for t in cands])
        best = np.flatnonzero(scores >= scores.max() - 1e-12)
        return float(cands[best[len(best) // 2]]), "max_macro_recall"
    cap = value
    neg = accept[y == OTHER]
    for t in cands:  # ascending: the smallest tau meeting the cap keeps the most targets
        if (neg >= t).mean() <= cap:
            return float(t), f"val_fpr_le_{cap}"
    return float(cands[-1]), "cap_unreachable"


def oracle_tau(S: np.ndarray, y: np.ndarray) -> float:
    """Threshold minimising (FNR + FPR) / 2 on the given data itself (an optimistic upper bound)."""
    cls, accept = S.argmax(1), S.max(1)
    tgt, oth = y < OTHER, y == OTHER
    best_tau, best_err = 0.5, np.inf
    cands = _candidates(accept)
    errors = []
    for t in cands:
        pred = np.where(accept >= t, cls, OTHER)
        errors.append(((pred[tgt] == OTHER).mean() + (pred[oth] < OTHER).mean()) / 2)
    errors = np.array(errors)
    best = np.flatnonzero(errors <= errors.min() + 1e-12)
    return float(cands[best[len(best) // 2]])


def compute_metrics(S: np.ndarray, y: np.ndarray, tau: float, subclass: list[str],
                    unseen: set[str] | frozenset = frozenset()) -> dict:
    pred, accept = decide(S, tau)
    tgt, oth = y < OTHER, y == OTHER
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (y, pred), 1)
    row_sum, col_sum = cm.sum(1), cm.sum(0)
    sub = np.array(subclass)
    seen_mask = np.array([s not in unseen for s in subclass])
    nan = float("nan")

    def rate(mask_sel, hit):
        return float(hit[mask_sel].mean()) if mask_sel.any() else nan

    accepted = pred < OTHER
    out = {
        "n": int(len(y)), "n_laptop": int(row_sum[0]), "n_book": int(row_sum[1]), "n_other": int(row_sum[2]),
        "accuracy": float((pred == y).mean()), "macro_recall": _macro_recall(y, pred),
        "recall_laptop": float(cm[0, 0] / row_sum[0]) if row_sum[0] else nan,
        "recall_book": float(cm[1, 1] / row_sum[1]) if row_sum[1] else nan,
        "recall_other": float(cm[2, 2] / row_sum[2]) if row_sum[2] else nan,
        "precision_laptop": float(cm[0, 0] / col_sum[0]) if col_sum[0] else nan,
        "precision_book": float(cm[1, 1] / col_sum[1]) if col_sum[1] else nan,
        "fnr": rate(tgt, pred == OTHER),                         # target rejected as "other"
        "swap_rate": rate(tgt, (pred != y) & (pred != OTHER)),   # laptop <-> book
        "fpr": rate(oth, accepted),                              # other accepted as a target
        "fpr_seen": rate(oth & seen_mask, accepted),
        "fpr_unseen": rate(oth & ~seen_mask, accepted) if unseen else nan,
    }
    if tgt.any() and oth.any():
        out["auroc"] = float(roc_auc_score(tgt.astype(int), accept))
        out["ap"] = float(average_precision_score(tgt.astype(int), accept))
    else:
        out["auroc"] = out["ap"] = nan
    out["confusion"] = json.dumps(cm.tolist())
    out["fpr_by_subclass"] = json.dumps({s: float(accepted[oth & (sub == s)].mean())
                                         for s in sorted(set(sub[oth]))})
    return out


class Recorder:
    """Collects metric rows and per-image predictions, then writes them as CSV."""

    def __init__(self, unseen: set[str] | frozenset = frozenset()):
        self.unseen = set(unseen)
        self.rows: list[dict] = []
        self.preds: list[dict] = []

    def add(self, *, stage: str, model: str, method: str, seed: int, S: np.ndarray,
            records: list[Record], splits: list[str], extra: dict | None = None,
            fixed_taus: dict[str, float] | None = None) -> None:
        """`S` is [n_records, 2] target-class scores (NaN rows are ignored).

        `fixed_taus` adds operating points with a given constant threshold (for example the
        0.05 group threshold used with the zero-shot detector), e.g. {"det005": 0.05}.
        """
        y = np.array([r.y for r in records])
        sub = [r.subclass for r in records]
        split_arr = np.array(splits)
        scored = ~np.isnan(S).any(1)
        val = (split_arr == "val") & scored
        pred_cols: dict[str, np.ndarray] = {}
        has_other_val = bool((y[val] == OTHER).any())
        ops = [op for op in OP_POINTS if has_other_val or op not in OPS_NEEDING_OTHER]
        for op in ops:
            tau, note = choose_tau(S[val], y[val], op)
            full_pred = np.full(len(records), OTHER)
            full_pred[scored] = decide(S[scored], tau)[0]
            pred_cols[op] = full_pred
            for split in EVAL_SPLITS:
                idx = np.flatnonzero((split_arr == split) & scored)
                if not len(idx):
                    continue
                m = compute_metrics(S[idx], y[idx], tau, [sub[i] for i in idx], self.unseen)
                self.rows.append({"stage": stage, "model": model, "method": method, "seed": seed,
                                  "split": split, "op_point": op, "tau": tau, "tau_rule": note,
                                  **m, **(extra or {})})
        for op, tau in (fixed_taus or {}).items():
            full_pred = np.full(len(records), OTHER)
            full_pred[scored] = decide(S[scored], tau)[0]
            pred_cols[op] = full_pred
            for split in EVAL_SPLITS:
                idx = np.flatnonzero((split_arr == split) & scored)
                if not len(idx):
                    continue
                m = compute_metrics(S[idx], y[idx], tau, [sub[i] for i in idx], self.unseen)
                self.rows.append({"stage": stage, "model": model, "method": method, "seed": seed,
                                  "split": split, "op_point": op, "tau": tau, "tau_rule": "fixed_threshold",
                                  **m, **(extra or {})})
        test = (split_arr == "test") & scored
        if (y[test] < OTHER).any() and (y[test] == OTHER).any():
            # Best-case threshold chosen on the test split itself: removes threshold-selection
            # noise when comparing models, but is optimistic and not deployable.
            tau = oracle_tau(S[test], y[test])
            full_pred = np.full(len(records), OTHER)
            full_pred[scored] = decide(S[scored], tau)[0]
            pred_cols["oracle"] = full_pred
            idx = np.flatnonzero(test)
            m = compute_metrics(S[idx], y[idx], tau, [sub[i] for i in idx], self.unseen)
            self.rows.append({"stage": stage, "model": model, "method": method, "seed": seed, "split": "test",
                              "op_point": "oracle", "tau": tau, "tau_rule": "test_oracle_min_balanced_error",
                              **m, **(extra or {})})
        for split in EVAL_SPLITS:
            for i in np.flatnonzero((split_arr == split) & scored):
                rec = records[i]
                self.preds.append({
                    "stage": stage, "model": model, "method": method, "seed": seed, "split": split,
                    "path": rec.path, "true": rec.label, "subclass": rec.subclass, "source": rec.source,
                    "score_laptop": round(float(S[i, 0]), 5), "score_book": round(float(S[i, 1]), 5),
                    **{f"pred_{op}": CLASSES[pred_cols[op][i]] for op in pred_cols}})

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = pd.DataFrame(self.rows)
        metrics.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(self.preds).to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
        if metrics.empty:
            return
        numeric = [c for c in metrics.select_dtypes("number").columns if c not in {"seed"}]
        keys = ["stage", "model", "method", "split", "op_point"]
        grouped = metrics.groupby(keys)[numeric]
        summary = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=0).add_suffix("_std"))
        summary["seeds"] = metrics.groupby(keys)["seed"].nunique()
        summary.reset_index().to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
