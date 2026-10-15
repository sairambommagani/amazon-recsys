"""Recommendation API for Amazon Reviews 2023 (Video Games).

    uvicorn amzrec.api:app --port 8007          (then open http://localhost:8007/docs)

GET  /users/sample                        real user ids to try
GET  /users/{user_id}/recommendations     top 10 next products with reasons + new products for this user
POST /next-items                          a basket / session of products -> what to buy next (no account needed)
GET  /items/{item_id}                     product details
GET  /items/{item_id}/similar             bought together (item-kNN), bought next (SASRec), similar description (text)
GET  /search?q=...                        search products by meaning, optionally personalised for a user
GET  /model-info, /health
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .als import ImplicitALS
from .pipeline import fuse
from .sasrec import SASRec
from .text import load_encoder

ART = Path(os.environ.get("AMZ_ARTIFACTS", "artifacts"))
N = 100


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


class Engine:
    def __init__(self, d: Path):
        if not (d / "config.json").exists():
            raise FileNotFoundError(f"{d}/config.json missing - run `python -m amzrec.train` first")
        self.cfg = json.loads((d / "config.json").read_text())
        self.metrics = json.loads((d / "metrics.json").read_text())
        self.items = pd.read_parquet(d / "items.parquet")
        self.n = len(self.items)
        self.item_index = pd.Index(self.items["item_id"])
        self.E = np.load(d / "E.npy").astype(np.float32)
        self.V = np.load(d / "V_als.npy")
        self.S = sp.load_npz(d / "knn.npz").tocsr()
        self.has_cf = self.items["has_cf"].to_numpy()
        self.is_new = self.items["is_new"].to_numpy()
        self.YtY = self.V[self.has_cf].T @ self.V[self.has_cf]
        self.pop = self.items["pop_recent"].to_numpy().astype(np.float32)
        self.pop_rank = np.empty(self.n, dtype=np.int64)
        self.pop_rank[np.argsort(-self.pop, kind="stable")] = np.arange(1, self.n + 1)
        sc = self.cfg["sasrec"]
        self.L = sc["L"]
        self.sas = {}
        for name, text in (("sasrec", None), ("sasrec_text", self.E)):
            m = SASRec(self.n, d=sc["d"], L=self.L, text=text)
            m.load_state_dict(torch.load(d / f"{name}.pt", map_location="cpu"))
            m.eval()
            self.sas[name] = (m, m.product_vectors())
        h = pd.read_parquet(d / "history.parquet")                        # sorted by user, then day
        self.h_item, self.h_day = h["i"].to_numpy(), h["day"].to_numpy()
        self.h_rating, self.h_pos = h["rating"].to_numpy(), h["pos"].to_numpy()
        n_users = int(h["u"].max()) + 1
        self.h_ptr = np.zeros(n_users + 1, dtype=np.int64)
        np.add.at(self.h_ptr, h["u"].to_numpy() + 1, 1)
        self.h_ptr = np.cumsum(self.h_ptr)
        self.user_ids = pd.read_parquet(d / "users.parquet")["user_id"].astype(str).to_numpy()
        self.user_index = pd.Index(self.user_ids)
        self.dir, self._enc = d, None

    @property
    def encoder(self):
        if self._enc is None:
            self._enc = load_encoder(self.dir / "encoder.joblib")
        return self._enc

    # ---- helpers
    def card(self, i: int, **extra) -> dict:
        r = self.items.iloc[i]
        cat = r["categories"].split(" > ")
        return {"item_id": r["item_id"], "title": r["title"], "category": cat[1] if len(cat) > 1 else (cat[0] or None),
                "store": r["store"] or None, "price": None if pd.isna(r["price"]) else float(r["price"]),
                "average_rating": float(r["average_rating"]), **extra}

    def item(self, item_id: str) -> int:
        k = self.item_index.get_indexer([item_id])[0]
        if k < 0:
            raise HTTPException(404, "item_id (parent_asin) not found in the 5-core catalogue")
        return int(k)

    def user(self, user_id: str) -> int:
        k = self.user_index.get_indexer([user_id])[0]
        if k < 0:
            raise HTTPException(404, "user_id not found (use GET /users/sample for valid ids)")
        return int(k)

    def history(self, u: int):
        a, b = self.h_ptr[u], self.h_ptr[u + 1]
        return self.h_item[a:b], self.h_day[a:b], self.h_rating[a:b], self.h_pos[a:b]

    def seq_vector(self, name: str, items: np.ndarray) -> np.ndarray:
        m, _ = self.sas[name]
        s = np.zeros((1, self.L), dtype=np.int64)
        last = np.asarray(items, dtype=np.int64)[-self.L:]
        if len(last):
            s[0, -len(last):] = last + 1
        return m.user_vectors(s)[0]

    # ---- core
    def recommend(self, items: np.ndarray, days: Optional[np.ndarray], liked: np.ndarray, exclude: np.ndarray,
                  k: int, k_new: int = 5) -> dict:
        """items: purchase sequence in time order; liked: boolean flag per purchase (4-5 stars)."""
        items = np.asarray(items, dtype=np.int64)
        liked_items = items[np.asarray(liked, bool)] if len(items) else items
        a = self.cfg["als"]
        w = None
        if days is not None and a.get("half_life") and len(liked_items):
            w = ImplicitALS.decay(self.cfg["last_day"] - np.asarray(days)[np.asarray(liked, bool)], a["half_life"])
        x = ImplicitALS.fold_in(liked_items[-500:], self.V, a["reg"], a["alpha"], self.YtY,
                                None if w is None else w[-500:])
        recent = liked_items[-20:]
        hk = np.zeros(self.n, np.float32)
        if len(recent):
            wk = 0.9 ** np.arange(len(recent))[::-1]
            row = sp.csr_matrix((wk, (np.zeros(len(recent), int), recent)), shape=(1, self.n))
            hk = (row @ self.S).toarray().ravel().astype(np.float32)        # sparse x sparse: stays small
        prof = _unit(self.E[liked_items[-50:]].mean(0)) if len(liked_items) else np.zeros(self.E.shape[1], np.float32)
        hs = {nm: self.seq_vector(nm, items) for nm in self.sas}
        ok = np.ones(self.n, bool)
        ok[exclude] = False

        def lists(c, scores):
            return {nm: c[np.argsort(-s, kind="stable")[:N]][None, :] for nm, s in scores.items()}

        c = np.flatnonzero(ok & self.has_cf)
        scores = {"popularity": self.pop[c]}
        if len(items):
            scores.update({"item_knn": hk[c], "als": self.V[c] @ x, "content": self.E[c] @ prof,
                           "sasrec": self.sas["sasrec"][1][c] @ hs["sasrec"],
                           "sasrec_text": self.sas["sasrec_text"][1][c] @ hs["sasrec_text"]})
        L_ = lists(c, scores)
        wf = self.cfg["warm_fusion"] if len(items) else {"weights": {"popularity": 1.0}, "c": 20}
        top = [int(i) for i in fuse(L_, wf["weights"], wf["c"], k)[0] if i >= 0]
        seq_top = set(L_.get("sasrec_text", np.zeros((1, 0)))[0][:20].tolist())
        new = np.flatnonzero(ok & self.is_new & ~self.has_cf)
        new_top = []
        if len(new) and len(items):
            Ln = lists(new, {"content": self.E[new] @ prof, "text_to_factors": self.V[new] @ x,
                             "sasrec_text": self.sas["sasrec_text"][1][new] @ hs["sasrec_text"]})
            wc = self.cfg["cold_fusion"]
            new_top = [int(i) for i in fuse(Ln, wc["weights"], wc["c"], k_new)[0] if i >= 0]
        return {"recommendations": [self.card(i, reasons=self.reasons(i, liked_items, recent, i in seq_top)) for i in top],
                "new_for_you": [self.card(i, reasons=self.reasons(i, liked_items, recent, False, new=True)) for i in new_top]}

    def reasons(self, i: int, liked_items: np.ndarray, recent: np.ndarray, in_seq: bool, new: bool = False) -> list[str]:
        out = []
        if new:
            first = pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(self.items.iloc[i]["first_day"]))
            out.append(f"new product (first review {first.date()}), picked from its description - almost no reviews yet")
        if len(recent) and not new:
            co = np.asarray(self.S[recent][:, i].todense()).ravel()
            j = int(np.argmax(co))
            if co[j] > 0:
                out.append(f"people who bought '{self.items.iloc[recent[j]]['title'][:60]}' also bought this")
        if in_seq:
            out.append("fits the order of your recent purchases (sequence model)")
        if len(liked_items):
            tx = self.E[liked_items[-50:]] @ self.E[i]
            j = int(np.argmax(tx))
            if tx[j] >= 0.5 or not out:
                out.append(f"similar to '{self.items.iloc[liked_items[-50:][j]]['title'][:60]}' (text match {tx[j]:.2f})")
        if not new and self.pop_rank[i] <= 100:
            out.append(f"popular in the last 90 days (#{self.pop_rank[i]})")
        return out


app = FastAPI(title="Amazon Reviews 2023 (Video Games) - Recommendation API", version="1.0",
              description="Sequence model (SASRec + product text), ALS, item-kNN, content and popularity, fused with weights "
                          "tuned on a validation period. Recommends products the user is likely to buy and rate 4-5 stars; "
                          "new products are recommended from their description.")
_cache: dict = {}


def eng() -> Engine:
    if "e" not in _cache:
        try:
            _cache["e"] = Engine(ART)
        except FileNotFoundError as e:
            raise HTTPException(503, str(e))
    return _cache["e"]


@app.get("/health")
def health():
    return {"status": "trained" if (ART / "config.json").exists() else "not trained"}


@app.get("/model-info")
def model_info():
    return eng().metrics


@app.get("/users/sample")
def users_sample(n: int = Query(5, ge=1, le=50), seed: int = 0):
    """Users with 10-60 purchases and a purchase in the last year of the data."""
    e = eng()
    cnt = np.diff(e.h_ptr)
    last_day = e.h_day[np.maximum(e.h_ptr[1:] - 1, 0)]
    pool = np.flatnonzero((cnt >= 10) & (cnt <= 60) & (last_day > e.cfg["last_day"] - 365))
    if len(pool) == 0:
        pool = np.flatnonzero(cnt > 0)
    pick = np.random.default_rng(seed).choice(pool, min(n, len(pool)), replace=False)
    return [{"user_id": e.user_ids[u], "purchases": int(cnt[u])} for u in pick]


@app.get("/users/{user_id}/recommendations")
def user_recs(user_id: str, k: int = Query(10, ge=1, le=50)):
    e = eng()
    u = e.user(user_id)
    items, days, ratings, liked = e.history(u)
    recent = [e.card(int(i), stars=int(r), date=str((pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(d))).date()))
              for i, d, r in list(zip(items, days, ratings))[::-1][:5]]
    return {"user_id": user_id, "purchases": int(len(items)), "recent_purchases": recent,
            **e.recommend(items, days, liked.astype(bool), items, k)}


class Basket(BaseModel):
    item_ids: list[str] = Field(..., min_length=1, max_length=50, description="parent_asin values, oldest first")
    k: int = Field(10, ge=1, le=50)
    model_config = {"json_schema_extra": {"examples": [{"item_ids": ["B0C1RKG2PM"], "k": 10}]}}


@app.post("/next-items")
def next_items(body: Basket):
    """Recommendations from a list of products alone (a new visitor or the current session) - no retraining."""
    e = eng()
    idx = [e.item_index.get_indexer([x])[0] for x in body.item_ids]
    items = np.array([i for i in idx if i >= 0], dtype=np.int64)
    if len(items) == 0:
        raise HTTPException(404, "none of the item_ids are in the catalogue (try ids from /users/{id}/recommendations)")
    return {"used_items": [e.card(int(i)) for i in items],
            **e.recommend(items, None, np.ones(len(items), bool), items, body.k)}


@app.get("/items/{item_id}")
def item(item_id: str):
    e = eng()
    i = e.item(item_id)
    r = e.items.iloc[i]
    return {**e.card(i), "categories": r["categories"], "details": r["details"], "features": r["features"][:300],
            "likes_in_data": int(r["n_likes"]), "likes_last_90_days": int(r["pop_recent"]), "new": bool(r["is_new"])}


@app.get("/items/{item_id}/similar")
def similar(item_id: str, k: int = Query(8, ge=1, le=30)):
    e = eng()
    i = e.item(item_id)
    row = e.S[i]
    o = np.argsort(-row.data)[:k]
    together = [e.card(int(row.indices[j]), similarity=round(float(row.data[j]), 3)) for j in o]
    hs = e.seq_vector("sasrec_text", np.array([i]))
    s = e.sas["sasrec_text"][1] @ hs
    s[i] = -np.inf
    s[~e.has_cf] = -np.inf
    nxt = [e.card(int(j)) for j in np.argsort(-s)[:k]]
    tx = e.E @ e.E[i]
    tx[i] = -1
    text = [e.card(int(j), similarity=round(float(tx[j]), 3)) for j in np.argsort(-tx)[:k]]
    return {**e.card(i), "bought_together": together, "bought_next": nxt, "similar_description": text}


@app.get("/search")
def search(q: str = Query(..., min_length=2, examples=["open world RPG with dragons"]), user_id: Optional[str] = None,
           k: int = Query(10, ge=1, le=50)):
    """Search by meaning. With user_id, the text matches are re-ranked by the user's sequence model."""
    e = eng()
    qv = e.encoder.encode([q])[0]
    s = e.E @ qv
    o = np.argsort(-s)[:N]
    lists, weights = {"text": o[None, :]}, {"text": 1.0}
    if user_id:
        items = e.history(e.user(user_id))[0]
        hs = e.seq_vector("sasrec_text", items)
        lists["personal"] = o[np.argsort(-(e.sas["sasrec_text"][1][o] @ hs))][None, :]
        weights["personal"] = 0.5
    top = fuse(lists, weights, 10, k)[0]
    return {"query": q, "personalised_for": user_id,
            "results": [e.card(int(i), text_match=round(float(s[i]), 3)) for i in top if i >= 0]}
