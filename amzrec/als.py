"""Implicit-feedback ALS (Hu, Koren & Volinsky 2008) solved with conjugate gradient (Takacs et al. 2011).

Every (user, item) cell is a preference p in {0, 1} (1 = the user liked the item). Observed cells get
confidence 1 + alpha, the rest confidence 1. The loss

    sum_{u,i} c_ui (p_ui - x_u . y_i)^2 + reg (|x_u|^2 + |y_i|^2)

is minimised by alternating: with the item factors Y fixed, every user vector has a closed form

    (Y'Y + alpha Y_u'Y_u + reg I) x_u = (1 + alpha) Y_u' 1

Time decay (optional): with a half-life h, a like that is `age` days old gets confidence
1 + alpha * 0.5 ** (age / h) instead of 1 + alpha, so recent tastes count more (h = None: no decay).

Solving that k x k system for 2M Yelp users one by one in Python is far too slow, so all users of a
chunk are solved at once with a few conjugate-gradient steps, warm-started from the previous
iteration. Only sparse-times-dense products and row-wise dot products are needed, so it is fast in
plain numpy/scipy and needs no compiled extension (important on Windows).
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp


class ImplicitALS:
    def __init__(self, factors: int = 64, reg: float = 1.0, alpha: float = 10.0, iters: int = 12,
                 cg_steps: int = 3, seed: int = 0, chunk_nnz: int = 400_000, half_life: float | None = None):
        self.factors, self.reg, self.alpha, self.iters, self.half_life = factors, reg, alpha, iters, half_life
        self.cg_steps, self.seed, self.chunk_nnz = cg_steps, seed, chunk_nnz

    def config(self) -> dict:
        return {"factors": self.factors, "reg": self.reg, "alpha": self.alpha, "iters": self.iters,
                "half_life": self.half_life}

    @staticmethod
    def decay(age_days, half_life):
        """Weight of a like that is `age_days` old."""
        age = np.asarray(age_days, dtype=np.float32)
        return np.ones_like(age) if not half_life else (0.5 ** (age / np.float32(half_life))).astype(np.float32)

    # ---- one half-step: solve all rows of X against fixed Y, starting from P (updated in place)
    def _half(self, X: sp.csr_matrix, Y: np.ndarray, P: np.ndarray) -> None:
        YtY = (Y.T @ Y).astype(np.float32)
        lam, a = np.float32(self.reg), np.float32(self.alpha)
        nnz_per_row = np.diff(X.indptr)
        r0, n = 0, X.shape[0]
        while r0 < n:
            # rows [r0, r1) with at most chunk_nnz entries
            r1 = int(np.searchsorted(X.indptr, X.indptr[r0] + self.chunk_nnz, side="right")) - 1
            r1 = min(max(r1, r0 + 1), n)
            sub = X[r0:r1]
            cols = sub.indices
            rows = np.repeat(np.arange(r1 - r0), nnz_per_row[r0:r1])
            Yc = Y[cols]                                                     # (nnz, k)
            shape = sub.shape

            w = sub.data                                                     # like weights (1 = no decay)

            def A(Z):
                t = np.einsum("ij,ij->i", Yc, Z[rows]) * w
                W = sp.csr_matrix((t, cols, sub.indptr), shape=shape)
                return Z @ YtY + lam * Z + a * (W @ Y)

            b = sp.csr_matrix((np.ones_like(w), cols, sub.indptr), shape=shape) @ Y + a * (sub @ Y)
            x = P[r0:r1]
            r = b - A(x)
            p = r.copy()
            rs = np.einsum("ij,ij->i", r, r)
            for _ in range(self.cg_steps):
                Ap = A(p)
                step = rs / np.maximum(np.einsum("ij,ij->i", p, Ap), 1e-12)
                x += step[:, None] * p
                r -= step[:, None] * Ap
                rs_new = np.einsum("ij,ij->i", r, r)
                p = r + (rs_new / np.maximum(rs, 1e-20))[:, None] * p
                rs = rs_new
            x[nnz_per_row[r0:r1] == 0] = 0.0
            P[r0:r1] = x
            r0 = r1

    def fit(self, X_age: sp.csr_matrix, log=None):
        """X_age: users x items, data = 1 + age in days of each like at the training cutoff."""
        X = sp.csr_matrix(X_age, dtype=np.float32, copy=True)
        X.data = self.decay(X.data - 1, self.half_life)
        X.sort_indices()
        XT = X.T.tocsr()
        rng = np.random.default_rng(self.seed)
        s = 0.1 / np.sqrt(self.factors)
        self.U = rng.normal(0, s, (X.shape[0], self.factors)).astype(np.float32)
        self.V = rng.normal(0, s, (X.shape[1], self.factors)).astype(np.float32)
        self.V[np.diff(XT.indptr) == 0] = 0.0
        for it in range(self.iters):
            t0 = time.time()
            self._half(X, self.V, self.U)
            self._half(XT, self.U, self.V)
            if log:
                log(f"    ALS iteration {it + 1}/{self.iters} ({time.time() - t0:.1f}s)")
        return self

    @staticmethod
    def fold_in(items: np.ndarray, V: np.ndarray, reg: float, alpha: float, YtY: np.ndarray | None = None,
                weights: np.ndarray | None = None) -> np.ndarray:
        """Exact user vector for a list of liked items (used by the API for any user, new or old)."""
        k = V.shape[1]
        if len(items) == 0:
            return np.zeros(k, dtype=np.float32)
        YtY = V.T @ V if YtY is None else YtY
        Yu = V[np.asarray(items)]
        w = np.ones(len(Yu), np.float32) if weights is None else np.asarray(weights, np.float32)
        A = YtY + alpha * ((Yu * w[:, None]).T @ Yu) + reg * np.eye(k, dtype=np.float32)
        return np.linalg.solve(A, ((1 + alpha * w)[:, None] * Yu).sum(axis=0)).astype(np.float32)
