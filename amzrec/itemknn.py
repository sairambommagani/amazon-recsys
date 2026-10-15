"""Item-kNN: "people who bought this also bought".

Two products are similar when the same users liked both (cosine similarity of their buyer sets). Only the
50 most similar products per product are kept (sparse). A user is scored from their 20 most recent likes,
recent ones weighted more.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize


class ItemKNN:
    def __init__(self, neighbours: int = 50, history: int = 20, recency: float = 0.9):
        self.k, self.history, self.recency = neighbours, history, recency

    def fit(self, X: sp.csr_matrix):
        Xb = X.copy().astype(np.float32)
        Xb.data[:] = 1.0
        Xn = normalize(Xb.T.tocsr(), axis=1)                           # items x users, unit rows
        S = (Xn @ Xn.T).tocsr()
        S.setdiag(0)
        S.eliminate_zeros()
        rows, cols, vals = [], [], []
        for i in range(S.shape[0]):                                    # keep the top-k neighbours
            a, b = S.indptr[i], S.indptr[i + 1]
            if a == b:
                continue
            c, v = S.indices[a:b], S.data[a:b]
            if len(v) > self.k:
                top = np.argpartition(-v, self.k)[:self.k]
                c, v = c[top], v[top]
            rows.append(np.full(len(c), i)); cols.append(c); vals.append(v)
        n = S.shape[0]
        self.S = sp.csr_matrix((np.concatenate(vals) if vals else [], (np.concatenate(rows) if rows else [],
                                np.concatenate(cols) if cols else [])), shape=(n, n), dtype=np.float32)
        return self

    def history_matrix(self, train_pos: pd.DataFrame, users: np.ndarray, n_items: int) -> sp.csr_matrix:
        sub = train_pos[train_pos["u"].isin(users)].sort_values("day").groupby("u").tail(self.history)
        rank = sub.groupby("u").cumcount(ascending=False).to_numpy()                  # 0 = most recent
        row = pd.Index(users).get_indexer(sub["u"].to_numpy())
        return sp.csr_matrix(((self.recency ** rank).astype(np.float32), (row, sub["i"].to_numpy())),
                             shape=(len(users), n_items))

    def score(self, H_rows: sp.csr_matrix) -> np.ndarray:
        return np.asarray((H_rows @ self.S).todense(), dtype=np.float32)
