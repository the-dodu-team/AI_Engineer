"""Heads on top of frozen, L2-normalised embeddings.

Each `fit_*` returns a function mapping embeddings to [n, 2] target-class scores
(laptop, book). When the training split has "other" images the head is trained as a
3-class model, otherwise as a 2-class model; either way only the target scores are
returned and the shared threshold rule decides between target and "other".
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

Scorer = Callable[[np.ndarray], np.ndarray]


def l2_normalize(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def fit_prototype(Xtr, ytr, Xva=None, yva=None, seed: int = 0) -> Scorer:
    """Class mean embeddings; the score is cosine similarity. No gradient training."""
    protos = np.stack([l2_normalize(Xtr[ytr == c].mean(0, keepdims=True))[0] for c in (0, 1)])
    return lambda X: X @ protos.T


def fit_knn(Xtr, ytr, Xva=None, yva=None, seed: int = 0, k: int = 5) -> Scorer:
    """Score = mean cosine similarity to the k nearest training images of each class.

    Learns nothing, and uses only laptop/book training images; a photo that is far from
    every training image gets a low score and is rejected as "other".
    """
    banks = [Xtr[ytr == c] for c in (0, 1)]

    def score(X):
        cols = []
        for bank in banks:
            sims = X @ bank.T
            top = np.partition(sims, -min(k, sims.shape[1]), axis=1)[:, -min(k, sims.shape[1]):]
            cols.append(top.mean(1))
        return np.stack(cols, 1)
    return score


def fit_mahalanobis(Xtr, ytr, Xva=None, yva=None, seed: int = 0) -> Scorer:
    """Score = -(squared Mahalanobis distance to the class mean) / dim.

    One shared, shrunk covariance (Ledoit-Wolf) is estimated from laptop/book training
    images only, a standard out-of-distribution baseline for pretrained features.
    """
    from sklearn.covariance import LedoitWolf

    means = np.stack([Xtr[ytr == c].mean(0) for c in (0, 1)])
    centered = np.concatenate([Xtr[ytr == c] - means[c] for c in (0, 1)])
    precision = LedoitWolf().fit(centered).precision_
    dim = Xtr.shape[1]

    def score(X):
        out = []
        for mean in means:
            d = X - mean
            out.append(-np.einsum("ij,jk,ik->i", d, precision, d) / dim)
        return np.stack(out, 1)
    return score


def _val_mask(yva, has_other):
    return np.ones(len(yva), dtype=bool) if has_other else yva < 2


def fit_logreg(Xtr, ytr, Xva, yva, seed: int = 0) -> Scorer:
    has_other = bool((ytr == 2).any())
    mask = _val_mask(yva, has_other)
    best, best_score = None, -1.0
    for C in (0.1, 1.0, 10.0, 100.0):
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=seed)
        clf.fit(Xtr, ytr)
        pred = clf.classes_[clf.predict_proba(Xva[mask]).argmax(1)] if mask.any() else None
        score = balanced_accuracy_score(yva[mask], pred) if mask.any() else 0.0
        if score > best_score:
            best, best_score = clf, score
    return lambda X: best.predict_proba(X)[:, [list(best.classes_).index(0), list(best.classes_).index(1)]]


def fit_mlp(Xtr, ytr, Xva, yva, seed: int = 0, hidden: int = 256, max_epochs: int = 300,
            patience: int = 25) -> Scorer:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    has_other = bool((ytr == 2).any())
    n_cls = 3 if has_other else 2
    mask = _val_mask(yva, has_other)
    counts = np.bincount(ytr, minlength=n_cls)[:n_cls].astype(float)
    weights = torch.tensor(counts.sum() / np.clip(counts, 1, None) / n_cls, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.GELU(), nn.Dropout(0.3), nn.Linear(hidden, n_cls))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr)
    Xv = torch.tensor(Xva[mask], dtype=torch.float32)
    yv = torch.tensor(yva[mask])
    gen = torch.Generator().manual_seed(seed)
    best_loss, best_state, wait = float("inf"), None, 0
    for _ in range(max_epochs):
        net.train()
        for idx in torch.randperm(len(Xt), generator=gen).split(32):
            opt.zero_grad()
            loss_fn(net(Xt[idx]), yt[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val_loss = loss_fn(net(Xv), yv).item() if len(Xv) else 0.0
        if val_loss < best_loss - 1e-4:
            best_loss, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()

    def score(X):
        with torch.no_grad():
            return torch.softmax(net(torch.tensor(X, dtype=torch.float32)), 1)[:, :2].numpy()
    return score


HEADS = {"prototype": fit_prototype, "knn": fit_knn, "mahalanobis": fit_mahalanobis,
         "logreg": fit_logreg, "mlp": fit_mlp}
