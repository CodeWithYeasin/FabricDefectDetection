"""v1 baseline: frozen ImageNet ResNet50 features + from-scratch second-order gradient boosting.

This is a cleaned, importable port of the NumPy boosting code from notebooks 07-10 (v1).
The maths is unchanged (Newton boosting with L2-regularised leaf weights, 10 percentile
thresholds per feature, 80% feature subsampling); the split search is vectorised so it runs in
minutes instead of hours.

Honest naming: in v1 the three "XGBoost / LightGBM / CatBoost" models were the same algorithm
with different (lambda, gamma) values. They are named here by what they are: GBDT-A/B/C.

Fix vs. v1: ensemble weights are fitted on a held-out *validation* split, never on the
training data (v1 notebook 10 fitted them on train+val, reaching 100% "accuracy").
"""
from __future__ import annotations

import numpy as np


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class _Node:
    __slots__ = ("feat", "thr", "left", "right", "value")

    def __init__(self):
        self.feat = self.thr = self.left = self.right = self.value = None


class NewtonTree:
    def __init__(self, max_depth=4, reg_lambda=1.0, gamma=0.0, min_child_weight=1.0,
                 feature_subsample=0.8, n_thresholds=10, rng=None):
        self.max_depth, self.lam, self.gamma = max_depth, reg_lambda, gamma
        self.mcw, self.fs, self.nt = min_child_weight, feature_subsample, n_thresholds
        self.rng = rng or np.random.default_rng()

    def _leaf(self, g, h):
        n = _Node()
        n.value = -g.sum() / (h.sum() + self.lam)
        return n

    def _best_split(self, X, g, h):
        n, d = X.shape
        feats = self.rng.choice(d, max(1, int(d * self.fs)), replace=False)
        Xs = X[:, feats]
        th = np.percentile(Xs, np.linspace(0, 100, self.nt), axis=0)  # (nt, f)
        M = Xs[:, None, :] <= th[None]  # (n, nt, f)
        GL = np.einsum("n,ntf->tf", g, M, optimize=True)
        HL = np.einsum("n,ntf->tf", h, M, optimize=True)
        G, H = g.sum(), h.sum()
        GR, HR = G - GL, H - HL
        cnt = M.sum(0)
        valid = (cnt > 0) & (cnt < n) & (HL >= self.mcw) & (HR >= self.mcw)
        gain = 0.5 * (GL**2 / (HL + self.lam) + GR**2 / (HR + self.lam) - G**2 / (H + self.lam)) - self.gamma
        gain = np.where(valid, gain, -np.inf)
        t, f = np.unravel_index(np.argmax(gain), gain.shape)
        if not np.isfinite(gain[t, f]) or gain[t, f] <= 0:
            return None, None
        return feats[f], th[t, f]

    def _build(self, X, g, h, depth):
        if depth >= self.max_depth or len(X) < 2:
            return self._leaf(g, h)
        f, t = self._best_split(X, g, h)
        if f is None:
            return self._leaf(g, h)
        node = _Node()
        node.feat, node.thr = int(f), float(t)
        m = X[:, f] <= t
        node.left = self._build(X[m], g[m], h[m], depth + 1)
        node.right = self._build(X[~m], g[~m], h[~m], depth + 1)
        return node

    def fit(self, X, g, h):
        self.root = self._build(X, g, h, 0)
        return self

    def predict(self, X):
        out = np.empty(len(X))
        stack = [(self.root, np.arange(len(X)))]
        while stack:
            node, idx = stack.pop()
            if node.value is not None:
                out[idx] = node.value
                continue
            m = X[idx, node.feat] <= node.thr
            stack.append((node.left, idx[m]))
            stack.append((node.right, idx[~m]))
        return out


class ScratchGBDT:
    def __init__(self, n_estimators=70, learning_rate=0.1, max_depth=4, reg_lambda=1.0, gamma=0.0, seed=0):
        self.K, self.lr, self.md, self.lam, self.gamma = n_estimators, learning_rate, max_depth, reg_lambda, gamma
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        self.n_classes = int(y.max()) + 1
        Y = np.eye(self.n_classes)[y]
        F = np.zeros((len(X), self.n_classes))
        self.trees = []
        for _ in range(self.K):
            P = softmax(F)
            G, H = P - Y, np.maximum(P * (1 - P), 1e-4)
            round_trees = []
            for k in range(self.n_classes):
                t = NewtonTree(self.md, self.lam, self.gamma, rng=self.rng).fit(X, G[:, k], H[:, k])
                F[:, k] += self.lr * t.predict(X)
                round_trees.append(t)
            self.trees.append(round_trees)
        return self

    def predict_proba(self, X):
        F = np.zeros((len(X), self.n_classes))
        for rt in self.trees:
            for k, t in enumerate(rt):
                F[:, k] += self.lr * t.predict(X)
        return softmax(F)


# v1 configurations (notebook 10): same learner, different regularisation
V1_CONFIGS = {
    "gbdt_a": dict(reg_lambda=1.0, gamma=0.0),   # v1 "XGBoost"
    "gbdt_b": dict(reg_lambda=0.1, gamma=0.0),   # v1 "LightGBM"
    "gbdt_c": dict(reg_lambda=2.0, gamma=0.1),   # v1 "CatBoost"
}


def fit_weights(prob_list, y, n_trials=5000, seed=0, metric="macro_f1"):
    """Dirichlet random search for convex ensemble weights on VALIDATION predictions."""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    best_w, best_s = np.ones(len(prob_list)) / len(prob_list), -1
    for i in range(n_trials):
        w = best_w if i == 0 else rng.dirichlet(np.ones(len(prob_list)))
        p = sum(wi * pi for wi, pi in zip(w, prob_list))
        pred = p.argmax(1)
        s = f1_score(y, pred, average="macro") if metric == "macro_f1" else (pred == y).mean()
        if s > best_s:
            best_s, best_w = s, w
    return best_w, best_s
