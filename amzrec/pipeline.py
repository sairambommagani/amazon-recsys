"""Splits, models, fusion and evaluation shared by MovieLens and Yelp.

Time-based protocol (no leakage):
  validation phase  train on everything before `val_start`, target = the next `val_days` days
                    -> choose the ALS settings, the fusion weights and the text->factor mapping
  test phase        train on everything before `test_start`, target = the last `test_days` days
                    -> the reported numbers (nothing is tuned here)

Two separate questions are evaluated:
  warm  rank every item that existed before the cutoff (the normal recommendation list)
  cold  rank only the NEW items that first appeared during the target window (new movies / new
        businesses). Collaborative filtering cannot score these at all: they have no interactions.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

K = 10
N_CAND = 100


# ----------------------------------------------------------------------------- metrics
def ranking_metrics(topk: np.ndarray, truth: list[np.ndarray], k: int = K) -> dict:
    """Recall@k (hits / min(#relevant, k)), NDCG@k and HitRate@k, averaged over users."""
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    rec, ndcg, hit = np.zeros(len(truth)), np.zeros(len(truth)), np.zeros(len(truth))
    for r, (row, rel) in enumerate(zip(topk[:, :k], truth)):
        h = np.isin(row, rel) & (row >= 0)
        n = min(len(rel), k)
        rec[r] = h.sum() / n
        ndcg[r] = (h * disc[:len(h)]).sum() / disc[:n].sum()
        hit[r] = h.any()
    return {"recall@10": rec, "ndcg@10": ndcg, "hit@10": hit}


def summarise(per_user: dict, topk: np.ndarray, n_candidates: int) -> dict:
    out = {m: round(float(v.mean()), 5) for m, v in per_user.items()}
    rec = topk[:, :K]
    out["coverage"] = round(len(np.unique(rec[rec >= 0])) / max(n_candidates, 1), 4)
    return out


# ----------------------------------------------------------------------------- split
@dataclass
class Split:
    train_end: int
    X: sp.csr_matrix               # liked before train_end (users x items), data = 1 + age in days
    seen: sp.csr_matrix            # any interaction before train_end
    warm: np.ndarray               # item had an interaction before train_end
    cold: np.ndarray               # item's first interaction is inside the target window
    pop: np.ndarray                # likes in the last 90 days before train_end
    warm_users: np.ndarray
    warm_truth: list
    cold_users: np.ndarray
    cold_truth: list
    n_hist: np.ndarray             # likes before train_end, per user
    train_pos: pd.DataFrame        # (u, i, day) likes before train_end


def build_split(inter: pd.DataFrame, n_users: int, n_items: int, first_day: np.ndarray, train_end: int,
                t0: int, t1: int, max_users: int, seed: int = 0, min_hist: int = 1) -> Split:
    tr = inter[inter["day"] < train_end]
    pos = tr[tr["pos"] == 1]
    last_like = pos.sort_values("day").drop_duplicates(["u", "i"], keep="last")
    X = sp.csr_matrix(((train_end - last_like["day"].to_numpy() + 1).astype(np.float32),     # 1 + age in days
                       (last_like["u"].to_numpy(), last_like["i"].to_numpy())), shape=(n_users, n_items))
    seen = sp.csr_matrix((np.ones(len(tr), np.int8), (tr["u"].to_numpy(), tr["i"].to_numpy())), shape=(n_users, n_items))
    warm = np.zeros(n_items, bool)
    warm[tr["i"].unique()] = True
    recent = pos[pos["day"] >= train_end - 90]
    pop = np.bincount(recent["i"], minlength=n_items).astype(np.float32)
    cold = (~warm) & (first_day >= t0) & (first_day <= t1)
    n_hist = np.diff(X.indptr)

    tg = inter[(inter["day"] >= t0) & (inter["day"] <= t1) & (inter["pos"] == 1)]
    tg = tg[n_hist[tg["u"].to_numpy()] >= min_hist]
    rng = np.random.default_rng(seed)
    out = {}
    for kind, mask in (("warm", warm), ("cold", cold)):
        t = tg[mask[tg["i"].to_numpy()]]
        users = t["u"].unique()
        if len(users) > max_users:
            users = np.sort(rng.choice(users, max_users, replace=False))
        t = t[t["u"].isin(users)]
        g = t.groupby("u")["i"].apply(lambda s: np.unique(s.to_numpy()))
        us, truth = [], []
        for u, items in g.items():
            items = np.setdiff1d(items, seen.indices[seen.indptr[u]:seen.indptr[u + 1]])
            if len(items):
                us.append(u)
                truth.append(items)
        out[kind] = (np.array(us, dtype=np.int64), truth)
    return Split(train_end, X, seen, warm, cold, pop, *out["warm"], *out["cold"], n_hist,
                 pos[["u", "i", "day"]])


def profiles(train_pos: pd.DataFrame, users: np.ndarray, E: np.ndarray, last: int = 50) -> np.ndarray:
    """Content profile of each user: the mean text vector of their `last` most recent likes."""
    sub = train_pos[train_pos["u"].isin(users)].sort_values("day").groupby("u").tail(last)
    row = pd.Index(users).get_indexer(sub["u"])
    M = sp.csr_matrix((np.ones(len(sub), np.float32), (row, sub["i"].to_numpy())), shape=(len(users), E.shape[0]))
    P = M @ E
    return (P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-8)).astype(np.float32)


def home_region(train: pd.DataFrame, users: np.ndarray, region_code: np.ndarray) -> np.ndarray:
    """Most frequent region among each user's past interactions (-1 = unknown)."""
    sub = train[train["u"].isin(users)]
    c = pd.DataFrame({"u": sub["u"].to_numpy(), "r": region_code[sub["i"].to_numpy()]}).value_counts()
    best = c.reset_index().sort_values("count", ascending=False).drop_duplicates("u").set_index("u")["r"]
    return best.reindex(users).fillna(-1).to_numpy(np.int64)


# ----------------------------------------------------------------------------- candidate scoring
def top_n(S: np.ndarray, n: int) -> np.ndarray:
    n = min(n, S.shape[1])
    part = np.argpartition(-S, n - 1, axis=1)[:, :n]
    rows = np.arange(S.shape[0])[:, None]
    order = np.argsort(-S[rows, part], axis=1, kind="stable")
    ids = part[rows, order]
    ids[~np.isfinite(S[rows, ids])] = -1
    return ids


def score_lists(users: np.ndarray, sp_: Split, models: dict, candidates: np.ndarray, region: np.ndarray | None,
                home: np.ndarray | None, n: int = N_CAND, batch: int = 256, exclude_seen: bool = True) -> dict:
    """For every model: the top-n candidate item ids per user (best first, -1 = padding).
    models: name -> function(rows, user_ids) returning scores over `candidates`."""
    out = {m: np.full((len(users), min(n, len(candidates))), -1, dtype=np.int64) for m in models}
    pos_of = np.full(sp_.X.shape[1], -1, dtype=np.int64)
    pos_of[candidates] = np.arange(len(candidates))
    for s in range(0, len(users), batch):
        rows = np.arange(s, min(s + batch, len(users)))
        ub = users[rows]
        mask = np.zeros((len(rows), len(candidates)), dtype=bool)
        if exclude_seen:
            seen = sp_.seen[ub].tocoo()
            p = pos_of[seen.col]
            ok = p >= 0
            mask[seen.row[ok], p[ok]] = True
        if region is not None:
            mask |= region[candidates][None, :] != home[rows][:, None]
        for name, fn in models.items():
            S = fn(rows, ub).astype(np.float32, copy=False)
            S[mask] = -np.inf
            ids = top_n(S, out[name].shape[1])
            out[name][rows] = np.where(ids >= 0, candidates[np.maximum(ids, 0)], -1)
    return out


def fuse(lists: dict, weights: dict, c: float, k: int = K) -> np.ndarray:
    """Weighted reciprocal-rank fusion: score(item) = sum_m w_m / (c + rank_m(item))."""
    rows, items, contrib = [], [], []
    for m, w in weights.items():
        if w == 0 or m not in lists:
            continue
        ids = lists[m]
        n_u, n = ids.shape
        r = np.repeat(np.arange(n_u), n)
        it = ids.ravel()
        ok = it >= 0
        rows.append(r[ok])
        items.append(it[ok])
        contrib.append((w / (c + np.tile(np.arange(n), n_u)))[ok])
    n_u = next(iter(lists.values())).shape[0]
    out = np.full((n_u, k), -1, dtype=np.int64)
    if not rows:
        return out
    rows, items, contrib = np.concatenate(rows), np.concatenate(items), np.concatenate(contrib)
    if len(items) == 0:                                              # no candidates at all
        return out
    base = int(items.max()) + 1
    key = rows.astype(np.int64) * base + items
    uniq, inv = np.unique(key, return_inverse=True)
    score = np.bincount(inv, contrib)
    ur, ui = uniq // base, uniq % base
    order = np.lexsort((-score, ur))
    ur, ui = ur[order], ui[order]
    start = np.searchsorted(ur, np.arange(n_u))
    end = np.searchsorted(ur, np.arange(n_u), side="right")
    for u in range(n_u):
        sel = ui[start[u]:min(end[u], start[u] + k)]
        out[u, :len(sel)] = sel
    return out


def tune_fusion(lists: dict, truth: list, grid: dict, base: str, log=print) -> tuple[dict, float, float]:
    """Grid search of fusion weights (base model weight fixed at 1) and the rank constant c.
    The base model alone (all other weights 0) is part of the grid, so fusion is never chosen
    unless it beats the base model on validation."""
    import itertools
    names = list(grid)
    best = (-1.0, None, None)
    for c in (5, 20, 60):
        for combo in itertools.product(*[grid[m] for m in names]):
            w = {base: 1.0, **dict(zip(names, combo))}
            v = ranking_metrics(fuse(lists, w, c), truth)["ndcg@10"].mean()
            if v > best[0]:
                best = (v, w, c)
    log(f"    best fusion on validation: weights {best[1]}, c={best[2]} (NDCG@10 {best[0]:.4f})")
    return best[1], best[2], best[0]


# ----------------------------------------------------------------------------- text -> collaborative factors
def fit_text_to_factors(E: np.ndarray, V: np.ndarray, items: np.ndarray, alpha: float):
    """Ridge regression from an item's text vector to its ALS factor vector, learned on items with
    enough likes. A brand-new item then gets a factor vector from its text alone, and can be scored
    for every user with the same dot product as old items."""
    from sklearn.linear_model import Ridge
    return Ridge(alpha=alpha, fit_intercept=True).fit(E[items], V[items])


def timer():
    t0 = time.time()
    return lambda: f"{time.time() - t0:.0f}s"
