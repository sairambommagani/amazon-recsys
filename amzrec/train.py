"""Train, tune and evaluate everything on Amazon Reviews 2023 (Video Games), then save the serving model.

    python -m amzrec.train --raw data\\raw            (add --fast for a quicker run)

Protocol (time-based, no leakage):
  validation phase  train on everything before `val_start`, target = the next `val_days` days
                    -> ALS setting, SASRec epochs (early stopping), fusion weights, text->factors mapping
  test phase        train on everything before `test_start`, target = the last `test_days` days
                    -> the reported numbers (nothing is tuned here)
A "hit" is a product the user bought AND rated 4-5 stars in the target period.

Old products (existed before the cutoff): popularity, item-kNN, ALS, content (MiniLM), SASRec, SASRec+text, fusion.
New products (first review inside the target period): random, content, text->ALS factors, SASRec+text, fusion.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .als import ImplicitALS
from .data import load
from .itemknn import ItemKNN
from .pipeline import build_split, fit_text_to_factors, fuse, profiles, ranking_metrics, score_lists, summarise
from .sasrec import SASRec, sequences, train_sasrec
from .text import make_encoder

# ALS settings tried on validation; the half-life (days) is the time decay (None = every purchase counts the same).
# The winning half-life is also used to weight SASRec's training loss, so both models favour recent tastes.
ALS_GRID = [{"alpha": 10.0, "reg": 20.0, "half_life": None}, {"alpha": 10.0, "reg": 20.0, "half_life": 365.0},
            {"alpha": 10.0, "reg": 20.0, "half_life": 90.0}]
WEIGHTS = [0, 0.25, 0.5, 1.0, 2.0]


def tune_fusion(lists: dict, truth: list, log, rounds: int = 2) -> tuple[dict, float, float]:
    """Coordinate ascent on reciprocal-rank-fusion weights: start from the best single model (weight 1),
    then try each other model's weight in turn, keeping a change only if validation NDCG@10 improves."""
    def ndcg(w, c):
        return float(ranking_metrics(fuse(lists, w, c), truth)["ndcg@10"].mean())
    single = {m: ndcg({m: 1.0}, 60) for m in lists}
    base = max(single, key=single.get)
    best_w, best_c, best = {base: 1.0}, 60.0, single[base]
    for c in (5.0, 20.0, 60.0):
        w = dict(best_w)
        v = ndcg(w, c)
        if v > best:
            best, best_c = v, c
    for _ in range(rounds):
        for m in lists:
            if m == base:
                continue
            for x in WEIGHTS:
                w = {**best_w, m: x}
                v = ndcg(w, best_c)
                if v > best + 1e-6:
                    best, best_w = v, w
    best_w = {m: x for m, x in best_w.items() if x > 0}
    log(f"    fusion: single models {({k: round(v, 4) for k, v in single.items()})} -> weights {best_w}, c={best_c} "
        f"(validation NDCG@10 {best:.4f})")
    return best_w, best_c, best


def cached_text_vectors(ds, enc, cutoff: int, cache: Path, log) -> np.ndarray:
    if enc.kind != "minilm":
        return enc.fit_encode(ds.texts(cutoff), log=log).astype(np.float32)
    path = cache / f"text_minilm_{len(ds.items)}_{cutoff}.npy"
    if path.exists():
        log(f"  text vectors loaded from cache ({path.name})")
        return np.load(path)
    E = enc.fit_encode(ds.texts(cutoff), log=log).astype(np.float32)
    np.save(path, E)
    return E


def run(ds, a, log) -> dict:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    inter = ds.inter
    n_users, n_items = len(ds.user_ids), len(ds.items)
    first = inter.groupby("i")["day"].min().reindex(np.arange(n_items)).fillna(10**9).to_numpy(np.int64)
    last = int(inter["day"].max())
    test0 = last - a.test_days + 1
    val0 = test0 - a.val_days
    log(f"5-core data: {len(inter):,} reviews ({inter['pos'].mean():.0%} rated 4-5), {n_users:,} users, {n_items:,} products, "
        f"{ds.date(inter['day'].min())} .. {ds.date(last)}")
    log(f"  validation = {ds.date(val0)} .. {ds.date(test0 - 1)}, test = {ds.date(test0)} .. {ds.date(last)}")

    log(f"encoding {n_items:,} product texts")
    enc = make_encoder(a.encoder, log)
    E = cached_text_vectors(ds, enc, val0, Path(a.cache), log)
    log(f"  encoder: {enc.kind}, {E.shape[1]} dimensions")

    def sasrec_model(cutoff, text):
        m = SASRec(n_items, d=a.dim, L=a.seq_len, text=E if text else None)
        seen = np.zeros(n_items + 1, bool); seen[0] = True
        seen[inter.loc[inter["day"] < cutoff, "i"].unique() + 1] = True
        m.seen.copy_(torch.from_numpy(seen))
        return m

    def train_seq(cutoff, half_life):
        users = inter.loc[inter["day"] < cutoff, "u"].unique()
        seq, pos, days = sequences(inter, users, cutoff, a.seq_len + 1, with_days=True)
        weight = None if not half_life else ImplicitALS.decay(np.maximum(cutoff - days, 0), half_life)
        return seq, pos, weight

    def fit_models(sp_, cutoff, als_cfg, sas_epochs=None, tune_users=None, tune_truth=None):
        """All models trained on data before `cutoff`."""
        m = {"als": ImplicitALS(factors=a.factors, iters=a.iters, **als_cfg).fit(sp_.X),
             "knn": ItemKNN().fit(sp_.X)}
        seq, pos, weight = train_seq(cutoff, als_cfg.get("half_life") if a.sasrec_decay else None)
        for name, text in (("sasrec", False), ("sasrec_text", True)):
            model = sasrec_model(cutoff, text)
            eval_fn = None
            if tune_users is not None:
                cands = np.flatnonzero(sp_.warm)
                s_in, _ = sequences(inter, tune_users, cutoff, a.seq_len)

                def eval_fn(mm, s_in=s_in, cands=cands):
                    Hs, Vs = mm.user_vectors(s_in), mm.product_vectors()[cands]
                    L_ = score_lists(tune_users, sp_, {"x": lambda r, u: Hs[r] @ Vs.T}, cands, None, None)
                    return ranking_metrics(L_["x"], tune_truth)["ndcg@10"].mean()
            log(f"  training {name} ({len(seq):,} user sequences)")
            model, ep, hist = train_sasrec(model, seq, pos, epochs=sas_epochs[name] if sas_epochs else a.max_epochs,
                                           eval_fn=eval_fn, patience=a.patience, log=log, weight=weight)
            m[name], m[name + "_epochs"] = model, ep
        return m

    def warm_lists(m, sp_, users, extra_users_seq=None):
        cands = np.flatnonzero(sp_.warm)
        Vc, Ec, popc = m["als"].V[cands], E[cands], sp_.pop[cands]
        P = profiles(sp_.train_pos, users, E)
        Hk = m["knn"].history_matrix(sp_.train_pos, users, n_items)
        s_in, _ = sequences(inter, users, sp_.train_end, a.seq_len)
        Hs = {k: m[k].user_vectors(s_in) for k in ("sasrec", "sasrec_text")}
        Vs = {k: m[k].product_vectors()[cands] for k in ("sasrec", "sasrec_text")}
        fns = {"popularity": lambda r, u: np.tile(popc, (len(r), 1)),
               "item_knn": lambda r, u: m["knn"].score(Hk[r])[:, cands],
               "als": lambda r, u: m["als"].U[u] @ Vc.T,
               "content": lambda r, u: P[r] @ Ec.T,
               "sasrec": lambda r, u: Hs["sasrec"][r] @ Vs["sasrec"].T,
               "sasrec_text": lambda r, u: Hs["sasrec_text"][r] @ Vs["sasrec_text"].T}
        return score_lists(users, sp_, fns, cands, None, None), len(cands)

    def cold_lists(m, sp_, users, Vhat_c, seed=0):
        cands = np.flatnonzero(sp_.cold)
        P = profiles(sp_.train_pos, users, E)
        s_in, _ = sequences(inter, users, sp_.train_end, a.seq_len)
        Hs = m["sasrec_text"].user_vectors(s_in)
        Vs = m["sasrec_text"].product_vectors()[cands]                 # new products: text part only
        rng = np.random.default_rng(seed)
        fns = {"random": lambda r, u: rng.random((len(r), len(cands))),
               "content": lambda r, u: P[r] @ E[cands].T,
               "text_to_factors": lambda r, u: m["als"].U[u] @ Vhat_c.T,
               "sasrec_text": lambda r, u: Hs[r] @ Vs.T}
        return score_lists(users, sp_, fns, cands, None, None), len(cands)

    def mapper(m, sp_, alpha):
        likes = np.bincount(sp_.X.indices, minlength=n_items)
        return fit_text_to_factors(E, m["als"].V, np.flatnonzero(sp_.warm & (likes >= a.min_item_likes)), alpha)

    # ======================================================== validation phase
    log("VALIDATION phase (tuning)")
    sv = build_split(inter, n_users, n_items, first, val0, val0, test0 - 1, a.max_users, seed=1)
    log(f"  {sv.X.nnz:,} training likes; {len(sv.warm_users):,} users to evaluate, {len(sv.cold_users):,} users who liked "
        f"new products ({sv.cold.sum():,} new products)")
    tu, tt = sv.warm_users[:a.tune_users], sv.warm_truth[:a.tune_users]
    best, best_cfg = -1.0, ALS_GRID[0]
    for cfg in (ALS_GRID[:1] if a.fast else ALS_GRID):
        als = ImplicitALS(factors=a.factors, iters=a.iters, **cfg).fit(sv.X)
        cands = np.flatnonzero(sv.warm)
        L_ = score_lists(tu, sv, {"als": lambda r, u: als.U[u] @ als.V[cands].T}, cands, None, None)
        v = ranking_metrics(L_["als"], tt)["ndcg@10"].mean()
        log(f"  ALS {cfg}: validation NDCG@10 {v:.4f}")
        if v > best:
            best, best_cfg = v, cfg
    mv = fit_models(sv, val0, best_cfg, tune_users=tu, tune_truth=tt)
    epochs = {k: max(1, mv[k + "_epochs"]) for k in ("sasrec", "sasrec_text")}
    log(f"  best epochs: {epochs}")
    lists, _ = warm_lists(mv, sv, tu)
    w_warm, c_warm, _ = tune_fusion(lists, tt, log)
    best_alpha, w_cold, c_cold = 10.0, {"sasrec_text": 1.0}, 20.0
    cu, ct = sv.cold_users[:a.tune_users], sv.cold_truth[:a.tune_users]
    if len(cu):
        bv = -1.0
        for alpha in (1.0, 10.0, 100.0, 1000.0):
            reg = mapper(mv, sv, alpha)
            cl, _ = cold_lists(mv, sv, cu, reg.predict(E[np.flatnonzero(sv.cold)]).astype(np.float32))
            v = ranking_metrics(cl["text_to_factors"], ct)["ndcg@10"].mean()
            if v > bv:
                bv, best_alpha, best_cl = v, alpha, cl
        best_cl.pop("random")
        w_cold, c_cold, _ = tune_fusion(best_cl, ct, log)
    del mv, lists

    # ======================================================== test phase
    log("TEST phase (last period, nothing tuned here)")
    st = build_split(inter, n_users, n_items, first, test0, test0, last, a.max_users, seed=2)
    log(f"  {st.X.nnz:,} training likes; {len(st.warm_users):,} test users, {len(st.cold_users):,} users who liked new "
        f"products ({st.cold.sum():,} new products)")
    mt = fit_models(st, test0, best_cfg, sas_epochs=epochs)
    lists, n_c = warm_lists(mt, st, st.warm_users)
    lists["fusion"] = fuse(lists, w_warm, c_warm)
    likes_train = np.bincount(st.X.indices, minlength=n_items)
    warm_ids = np.flatnonzero(st.warm)
    head = np.zeros(n_items, bool)
    head[warm_ids[np.argsort(-likes_train[warm_ids])[:max(1, len(warm_ids) // 5)]]] = True   # top 20% = "popular"
    res, per_user, by_pop = {}, {}, {}
    for name, L_ in lists.items():
        per_user[name] = ranking_metrics(L_, st.warm_truth)
        res[name] = summarise(per_user[name], L_, n_c)
        by_pop[name] = {}
        for label, mask in (("popular_products", head), ("long_tail_products", ~head)):
            tr = [t[mask[t]] for t in st.warm_truth]
            keep = np.array([len(t) > 0 for t in tr])
            if keep.any():
                by_pop[name][label] = round(float(ranking_metrics(L_[keep], [t for t in tr if len(t)])["recall@10"].mean()), 5)
        log(f"  old products  {name:12s} " + "  ".join(f"{k} {v}" for k, v in res[name].items()) + f"  recall by popularity {by_pop[name]}")
    nh = st.n_hist[st.warm_users]
    seg = {}
    for lo, hi, label in ((1, 4, "1-4 likes"), (5, 9, "5-9 likes"), (10, 19, "10-19 likes"), (20, 10**9, "20+ likes")):
        s = (nh >= lo) & (nh <= hi)
        seg[label] = {"users": int(s.sum()), **{k: (round(float(v["ndcg@10"][s].mean()), 5) if s.any() else None)
                                                for k, v in per_user.items()}}
    log(f"  NDCG@10 by user history: {json.dumps(seg)}")
    cold_res = {}
    reg = mapper(mt, st, best_alpha)
    if len(st.cold_users):
        Vhat_c = reg.predict(E[np.flatnonzero(st.cold)]).astype(np.float32)
        cl, n_cold = cold_lists(mt, st, st.cold_users, Vhat_c)
        cl["fusion"] = fuse({k: v for k, v in cl.items() if k != "random"}, w_cold, c_cold)
        for name, L_ in cl.items():
            cold_res[name] = summarise(ranking_metrics(L_, st.cold_truth), L_, n_cold)
            log(f"  new products  {name:15s} " + "  ".join(f"{k} {v}" for k, v in cold_res[name].items()))

    # ======================================================== serving artifacts
    log("saving the serving model")
    V = mt["als"].V.copy()
    V[~st.warm] = reg.predict(E[~st.warm]).astype(np.float32)
    items = ds.items.copy()
    likes = inter[inter["pos"] == 1]
    items["first_day"] = first
    items["n_likes"] = np.bincount(likes["i"], minlength=n_items)
    items["pop_recent"] = np.bincount(likes.loc[likes["day"] > last - 90, "i"], minlength=n_items)
    items["is_new"] = (first > last - a.new_days) & (first < 10**9)
    items["has_cf"] = st.warm
    items.to_parquet(out / "items.parquet", index=False)
    np.save(out / "E.npy", E.astype(np.float16))
    np.save(out / "V_als.npy", V)
    sp.save_npz(out / "knn.npz", mt["knn"].S)
    for k in ("sasrec", "sasrec_text"):
        torch.save(mt[k].state_dict(), out / f"{k}.pt")
    pd.DataFrame({"user_id": ds.user_ids}).to_parquet(out / "users.parquet", index=False)
    inter[["u", "i", "day", "rating", "pos"]].to_parquet(out / "history.parquet", index=False)
    enc.save(out / "encoder.joblib")
    cfg = {"als": {**mt["als"].config(), **best_cfg}, "warm_fusion": {"weights": w_warm, "c": c_warm},
           "cold_fusion": {"weights": w_cold, "c": c_cold}, "ridge_alpha": best_alpha, "encoder": enc.kind,
           "sasrec": {"d": a.dim, "L": a.seq_len, "epochs": epochs,
                      "loss_half_life": best_cfg.get("half_life") if a.sasrec_decay else None}, "last_day": last, "last_date": ds.date(last)}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    report = {"dataset": "Amazon Reviews 2023 - Video Games (5-core)", "eda": ds.eda,
              "validation_period": [ds.date(val0), ds.date(test0 - 1)], "test_period": [ds.date(test0), ds.date(last)],
              "text_encoder": enc.kind, "config": cfg,
              "old_products": {"test_users": len(st.warm_users), "candidates": int(n_c), "results": res,
                               "recall@10_popular_vs_long_tail": by_pop, "ndcg@10_by_history": seg},
              "new_products": {"test_users": len(st.cold_users), "new_products": int(st.cold.sum()), "results": cold_res}}
    (out / "metrics.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--encoder", default="auto", choices=["auto", "minilm", "tfidf"])
    ap.add_argument("--test-days", type=int, default=180)
    ap.add_argument("--val-days", type=int, default=180)
    ap.add_argument("--new-days", type=int, default=180)
    ap.add_argument("--factors", type=int, default=64)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--dim", type=int, default=64, help="SASRec hidden size")
    ap.add_argument("--seq-len", type=int, default=50)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--max-users", type=int, default=20000)
    ap.add_argument("--tune-users", type=int, default=5000)
    ap.add_argument("--min-item-likes", type=int, default=5)
    ap.add_argument("--no-sasrec-decay", dest="sasrec_decay", action="store_false",
                    help="do not weight SASRec's loss by purchase recency")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--eda-only", action="store_true", help="load the data, print the exploration summary and stop")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args(argv)
    if a.fast:
        a.max_epochs, a.max_users, a.iters = min(a.max_epochs, 8), min(a.max_users, 5000), 6
    t0 = time.time()
    log = lambda *x: print(f"[{time.time() - t0:5.0f}s]", *x, flush=True)
    raw, cache = a.raw, a.cache
    if a.synthetic:
        from scripts.make_synthetic import write
        raw = write(Path(a.out) / "_synthetic_raw")
        cache = Path(a.out) / "_synthetic_cache"
        a.cache = str(cache)
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    ds = load(raw, cache, log)
    e = ds.eda["all_data"]
    log(f"EDA: {e['reviews']:,} reviews, {e['users']:,} users ({e['users_with_1_review']:.0%} with 1 review), "
        f"{e['products_reviewed']:,} products; ratings {e['rating_share']}")
    if a.eda_only:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "eda.json").write_text(json.dumps(ds.eda, indent=1))
        print(json.dumps(ds.eda, indent=1))
        return ds.eda
    rep = run(ds, a, log)
    r = rep["old_products"]["results"]
    log("DONE: NDCG@10 " + ", ".join(f"{k} {v['ndcg@10']}" for k, v in r.items()) + f". Artifacts in {a.out}")
    return rep


if __name__ == "__main__":
    main()
