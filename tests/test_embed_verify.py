from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embed_verify.data import Record, make_splits  # noqa: E402
from embed_verify.evalcore import Recorder, choose_tau, compute_metrics, decide  # noqa: E402
from embed_verify.heads import HEADS, l2_normalize  # noqa: E402


def records(n_target=20, n_other=20):
    recs = []
    for label in ("laptop", "book"):
        for i in range(n_target):
            recs.append(Record(f"{label}{i}.jpg", label, label, "coco" if i % 2 else "oi", f"{label}{i}", "train"))
    for sub in ("bed", "table"):
        for i in range(n_other):
            recs.append(Record(f"{sub}{i}.jpg", "other", sub, "unknown", f"{sub}{i}", "other"))
    return recs


class SplitTests(unittest.TestCase):
    def test_deterministic_stratified_and_unseen(self):
        recs = records()
        a = make_splits(recs, 0, {"table"})
        self.assertEqual(a, make_splits(recs, 0, {"table"}))
        self.assertNotEqual(a, make_splits(recs, 1, {"table"}))
        for r, s in zip(recs, a):
            if r.subclass == "table":
                self.assertEqual(s, "test")
        self.assertEqual(set(a), {"train", "val", "test"})
        # Every image lands in exactly one split, and the shares follow the ratios.
        # By default "other" is used for testing only.
        count = lambda sp, sub: [s for r, s in zip(recs, sp) if r.subclass == sub]
        self.assertEqual(set(count(a, "bed")), {"test"})
        shared = make_splits(recs, 0, frozenset(), (0.6, 0.2, 0.2), other_in_val=True)
        self.assertEqual((count(shared, "bed").count("train"), count(shared, "bed").count("val"),
                          count(shared, "bed").count("test")), (0, 10, 10))
        trained = make_splits(recs, 0, frozenset(), (0.6, 0.2, 0.2), other_in_train=True)
        self.assertGreater([s for r, s in zip(recs, trained) if r.subclass == "bed"].count("train"), 0)
        for label in ("laptop", "book"):
            self.assertTrue(all(any(r.label == label and s == sp for r, s in zip(recs, a))
                                for sp in ("train", "val", "test")))


class MetricTests(unittest.TestCase):
    def test_decide_and_metrics(self):
        S = np.array([[.9, .1], [.2, .8], [.3, .2], [.7, .1]])
        y = np.array([0, 1, 2, 2])
        pred, _ = decide(S, 0.5)
        self.assertEqual(pred.tolist(), [0, 1, 2, 0])
        m = compute_metrics(S, y, 0.5, ["laptop", "book", "bed", "table"], {"table"})
        self.assertEqual(m["fnr"], 0.0)
        self.assertEqual(m["fpr"], 0.5)
        self.assertEqual(m["fpr_unseen"], 1.0)
        self.assertEqual(m["fpr_seen"], 0.0)
        self.assertGreater(m["auroc"], 0.5)

    def test_threshold_uses_validation_only_and_caps_fpr(self):
        rng = np.random.default_rng(0)
        S = np.concatenate([rng.uniform(.6, 1, (30, 2)), rng.uniform(0, .7, (30, 2))])
        y = np.array([0] * 15 + [1] * 15 + [2] * 30)
        tau, _ = choose_tau(S, y, "fpr1")
        pred, _ = decide(S, tau)
        self.assertLessEqual((pred[y == 2] < 2).mean(), 0.01)
        tau_macro, note = choose_tau(S, y, "macro")
        self.assertEqual(note, "max_macro_recall")
        self.assertGreater(tau_macro, 0.5)

    def test_quantile_threshold_needs_no_other(self):
        S = np.linspace(.1, .9, 20)[:, None] * np.ones((1, 2))
        y = np.zeros(20, dtype=int)
        tau, note = choose_tau(S, y, "q95")
        self.assertEqual(note, "accept_95pct_of_val_targets")
        self.assertAlmostEqual((S.max(1) >= tau).mean(), 0.95, delta=0.06)
        with self.assertRaises(ValueError):
            choose_tau(S, y, "macro")


class HeadTests(unittest.TestCase):
    def test_heads_separate_clusters(self):
        rng = np.random.default_rng(0)
        centers = np.eye(3, 16)
        X = np.concatenate([centers[c] + .1 * rng.normal(size=(40, 16)) for c in range(3)])
        y = np.repeat([0, 1, 2], 40)
        X = l2_normalize(X)
        idx = rng.permutation(len(y))
        tr, va = idx[:80], idx[80:]
        for name, fit in HEADS.items():
            S = fit(X[tr], y[tr], X[va], y[va], seed=0)(X)
            self.assertEqual(S.shape, (120, 2))
            self.assertGreater((S[y == 0, 0] > S[y == 0, 1]).mean(), .9, name)

    def test_positive_only_heads_reject_unseen_cluster(self):
        rng = np.random.default_rng(1)
        centers = np.eye(3, 32)
        X = l2_normalize(np.concatenate([centers[c] + .1 * rng.normal(size=(60, 32)) for c in range(3)]))
        y = np.repeat([0, 1, 2], 60)
        tr = np.flatnonzero(y < 2)[::2]
        for name in ("prototype", "knn", "mahalanobis"):
            S = HEADS[name](X[tr], y[tr], None, None, seed=0)(X)
            accept = S.max(1)
            self.assertGreater(accept[y < 2].mean(), accept[y == 2].mean(), name)
            tau, _ = choose_tau(S, y, "macro")
            pred, _ = decide(S, tau)
            self.assertLess((pred[y == 2] < 2).mean(), .1, name)

    def test_recorder_end_to_end(self):
        recs = records()
        splits = make_splits(recs, 0)
        y = np.array([r.y for r in recs])
        S = np.zeros((len(recs), 2))
        S[y == 0, 0], S[y == 1, 1] = .9, .9
        S[y == 2] = .1
        rec = Recorder()
        rec.add(stage="t", model="m", method="x", seed=0, S=S, records=recs, splits=splits)
        rows = {r["op_point"]: r for r in rec.rows if r["split"] == "test"}
        self.assertEqual(set(rows), {"q99", "q95", "oracle"})  # no "other" in val -> no macro/fpr rules
        self.assertEqual(rows["q99"]["fnr"], 0.0)
        self.assertEqual(rows["q99"]["fpr"], 0.0)
        self.assertEqual(rows["oracle"]["tau_rule"], "test_oracle_min_balanced_error")
        self.assertTrue(rec.preds)


if __name__ == "__main__":
    unittest.main()
