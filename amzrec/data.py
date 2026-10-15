"""Amazon Reviews 2023 (McAuley Lab): loading, exploration (EDA) and the 5-core filter.

Files (one JSON object per line, plain .jsonl or gzipped .jsonl.gz):
  Video_Games.jsonl          4.6M reviews (~2.7 GB): user_id, parent_asin, rating, timestamp (ms), title, text, ...
  meta_Video_Games.jsonl     137k products (~0.4 GB): parent_asin, title, store, categories, features,
                             description, details, price, average_rating, rating_number

Products are identified by `parent_asin` (all colour / edition variants of a product share it), as the
dataset authors recommend.

Memory: the review file is streamed line by line; ids become integers, timestamps become day numbers,
and review text is cut to 200 characters and stored in a separate Parquet file. Nothing close to
2.7 GB is ever in RAM.

5-core: users and products with at least 5 reviews, applied repeatedly until stable (removing a product
can push a user below 5 and so on). This is the standard benchmark setting for this dataset; the EDA
reports how much of the data it keeps.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import orjson
    _loads = orjson.loads
except ImportError:                                                     # pragma: no cover
    _loads = json.loads

DAY0 = pd.Timestamp("1970-01-01")
LIKE = 4                                                                # rating >= 4 = "liked"


@dataclass
class Dataset:
    inter: pd.DataFrame          # u, i, day, rating, pos  (5-core, sorted by user then day)
    items: pd.DataFrame          # one row per product in the 5-core catalogue
    user_ids: np.ndarray         # original user_id per user number
    texts: Callable[[int], list]  # product texts using only reviews written before the given day
    eda: dict
    item_label: str = "product"

    def date(self, day: int) -> str:
        return str((DAY0 + pd.Timedelta(days=int(day))).date())


def _find(raw: Path, meta: bool) -> Path:
    """The metadata file (meta_*.jsonl[.gz]) or the review file (any other *.jsonl[.gz])."""
    hits = sorted(p for p in Path(raw).rglob("*.jsonl*") if p.name.startswith("meta_") == meta)
    if not hits:
        raise FileNotFoundError(f"no {'meta_*' if meta else 'review'} .jsonl file under {raw}")
    return hits[0]


def _open(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb")


def _join(x, limit: int) -> str:
    if isinstance(x, list):
        x = " ".join(str(s) for s in x if s)
    return (str(x) if x else "")[:limit]


def _details(d) -> str:
    if not isinstance(d, dict):
        return ""
    keep = []
    for k, v in d.items():
        if k.lower() in ("best sellers rank", "customer reviews", "asin", "date first available", "item model number",
                         "is discontinued by manufacturer", "manufacturer"):
            continue
        if isinstance(v, (str, int, float)) and len(str(v)) < 60:
            keep.append(f"{k}: {v}")
        if len(keep) == 5:
            break
    return ", ".join(keep)


def _meta(raw: Path, cache: Path, log) -> pd.DataFrame:
    out = cache / "meta.parquet"
    if out.exists():
        return pd.read_parquet(out)
    src = _find(raw, meta=True)
    log(f"reading product metadata {src.name} ...")
    rows = []
    with _open(src) as f:
        for line in f:
            m = _loads(line)
            cats = m.get("categories") or []
            rows.append((m["parent_asin"], (m.get("title") or "")[:200], m.get("store") or "", m.get("main_category") or "",
                         " > ".join(cats) if isinstance(cats, list) else "", _join(m.get("features"), 600),
                         _join(m.get("description"), 600), _details(m.get("details")),
                         float(m["price"]) if isinstance(m.get("price"), (int, float)) else np.nan,
                         float(m.get("average_rating") or 0), int(m.get("rating_number") or 0)))
    meta = pd.DataFrame(rows, columns=["item_id", "title", "store", "main_category", "categories", "features",
                                       "description", "details", "price", "average_rating", "rating_number"])
    meta = meta.drop_duplicates("item_id").reset_index(drop=True)
    meta.to_parquet(out, index=False)
    log(f"  {len(meta):,} products")
    return meta


def _reviews(raw: Path, cache: Path, meta: pd.DataFrame, log, chunk: int = 500_000) -> None:
    out, out_text = cache / "reviews.parquet", cache / "review_text.parquet"
    if out.exists() and out_text.exists():
        return
    src = _find(raw, meta=False)
    log(f"converting {src.name} to Parquet (one-time, streamed; ~5-10 minutes for 4.6M reviews) ...")
    iidx = {a: k for k, a in enumerate(meta["item_id"])}
    users: dict[str, int] = {}
    buf = {k: [] for k in ("u", "i", "day", "rating", "verified", "title", "text")}
    w = wt = None
    n = skipped = bad = 0

    def flush():
        nonlocal w, wt
        t = pa.table({"u": np.array(buf["u"], np.int32), "i": np.array(buf["i"], np.int32),
                      "day": np.array(buf["day"], np.int32), "rating": np.array(buf["rating"], np.int8),
                      "verified": np.array(buf["verified"], np.int8)})
        tt = pa.table({"i": np.array(buf["i"], np.int32), "day": np.array(buf["day"], np.int32),
                       "rating": np.array(buf["rating"], np.int8), "title": buf["title"], "text": buf["text"]})
        w = w or pq.ParquetWriter(out.with_suffix(".tmp"), t.schema)
        wt = wt or pq.ParquetWriter(out_text.with_suffix(".tmp"), tt.schema)
        w.write_table(t)
        wt.write_table(tt)
        for v in buf.values():
            v.clear()

    with _open(src) as f:
        for line in f:
            try:
                r = _loads(line)
            except ValueError:                                       # e.g. a cut-off last line
                bad += 1
                continue
            i = iidx.get(r.get("parent_asin"))
            if i is None:
                skipped += 1
                continue
            buf["u"].append(users.setdefault(r["user_id"], len(users)))
            buf["i"].append(i)
            buf["day"].append(int(r["timestamp"]) // 86_400_000)
            buf["rating"].append(int(r["rating"]))
            buf["verified"].append(1 if r.get("verified_purchase") else 0)
            buf["title"].append((r.get("title") or "")[:100])
            buf["text"].append((r.get("text") or "")[:200].replace("\n", " "))
            n += 1
            if len(buf["u"]) >= chunk:
                flush()
                log(f"  {n:,} reviews converted")
    if buf["u"]:
        flush()
    w.close()
    wt.close()
    out.with_suffix(".tmp").rename(out)
    out_text.with_suffix(".tmp").rename(out_text)
    pd.DataFrame({"user_id": list(users)}).to_parquet(cache / "users_all.parquet", index=False)
    log(f"  {n:,} reviews, {len(users):,} users ({skipped:,} without product metadata, {bad} unreadable lines skipped)")


def k_core(df: pd.DataFrame, k: int = 5, log=print) -> pd.DataFrame:
    """Keep users and items with at least k interactions, repeated until nothing changes."""
    for it in range(50):
        uc = df["u"].map(df["u"].value_counts())
        ic = df["i"].map(df["i"].value_counts())
        keep = (uc >= k) & (ic >= k)
        if keep.all():
            break
        df = df[keep]
    log(f"  {k}-core after {it + 1} passes: {len(df):,} reviews, {df['u'].nunique():,} users, {df['i'].nunique():,} products")
    return df


def explore(df: pd.DataFrame, meta: pd.DataFrame) -> dict:
    per_user = df.groupby("u").size()
    per_item = df.groupby("i").size()
    q = lambda s: {p: int(s.quantile(p / 100)) for p in (50, 90, 99)}
    return {"reviews": int(len(df)), "users": int(per_user.size), "products_reviewed": int(per_item.size),
            "products_in_metadata": int(len(meta)),
            "date_range": [str((DAY0 + pd.Timedelta(days=int(df['day'].min()))).date()),
                           str((DAY0 + pd.Timedelta(days=int(df['day'].max()))).date())],
            "rating_share": {str(r): round(float(v), 4) for r, v in df["rating"].value_counts(normalize=True).sort_index().items()},
            "liked_share(rating>=4)": round(float((df["rating"] >= LIKE).mean()), 4),
            "verified_share": round(float(df["verified"].mean()), 4),
            "reviews_per_user_percentiles": q(per_user), "users_with_1_review": round(float((per_user == 1).mean()), 4),
            "users_with_5plus": round(float((per_user >= 5).mean()), 4),
            "reviews_per_product_percentiles": q(per_item)}


def load(raw, cache, log=print, k: int = 5) -> Dataset:
    raw, cache = Path(raw), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    meta = _meta(raw, cache, log)
    _reviews(raw, cache, meta, log)
    core_path = cache / f"core{k}.parquet"
    eda_path = cache / "eda.json"
    if core_path.exists() and eda_path.exists():
        core = pd.read_parquet(core_path)
        eda = json.loads(eda_path.read_text())
    else:
        allr = pd.read_parquet(cache / "reviews.parquet")
        allr = allr.sort_values("day", kind="stable").drop_duplicates(["u", "i"], keep="first")
        eda = {"all_data": explore(allr, meta)}
        core = k_core(allr, k, log)
        eda[f"{k}_core"] = explore(core, meta)
        eda[f"{k}_core"]["share_of_reviews_kept"] = round(len(core) / len(allr), 4)
        eda[f"{k}_core"]["share_of_users_kept"] = round(core["u"].nunique() / allr["u"].nunique(), 4)
        del allr
        core.to_parquet(core_path, index=False)
        eda_path.write_text(json.dumps(eda, indent=1))
    # compact numbering for the 5-core catalogue
    items_idx = np.sort(core["i"].unique())
    users_idx = np.sort(core["u"].unique())
    new_i = pd.Series(np.arange(len(items_idx), dtype=np.int32), index=items_idx)
    new_u = pd.Series(np.arange(len(users_idx), dtype=np.int32), index=users_idx)
    inter = pd.DataFrame({"u": new_u.reindex(core["u"]).to_numpy(), "i": new_i.reindex(core["i"]).to_numpy(),
                          "day": core["day"].to_numpy(), "rating": core["rating"].to_numpy()})
    inter["pos"] = (inter["rating"] >= LIKE).astype(np.int8)
    inter = inter.sort_values(["u", "day"], kind="stable").reset_index(drop=True)
    items = meta.iloc[items_idx].reset_index(drop=True)
    user_ids = pd.read_parquet(cache / "users_all.parquet")["user_id"].to_numpy()[users_idx]

    def texts(cutoff_day: int) -> list[str]:
        """Product text: title, category, store, details, then what buyers said (titles of the two most
        recent liked reviews written before cutoff_day), features and description."""
        f = pq.ParquetFile(cache / "review_text.parquet")
        best = None
        keep = set(items_idx.tolist())
        for g in range(f.num_row_groups):
            t = f.read_row_group(g, columns=["i", "day", "rating", "title"]).to_pandas()
            t = t[(t["day"] < cutoff_day) & (t["rating"] >= LIKE) & t["i"].isin(keep) & (t["title"].str.len() > 3)]
            if best is not None:
                t = pd.concat([best, t], ignore_index=True)
            best = t.sort_values(["i", "day"]).groupby("i", sort=False).tail(2)
        said = best.groupby("i")["title"].apply(lambda s: "; ".join(s[::-1])) if best is not None else pd.Series(dtype=str)
        said = said.reindex(items_idx).fillna("").to_numpy()
        out = []
        for row, s in zip(items.itertuples(index=False), said):
            parts = [row.title, f"Category: {row.categories}" if row.categories else "", f"Store: {row.store}" if row.store else "",
                     row.details, f"Buyers say: {s}" if s else "", row.features, row.description]
            out.append(". ".join(p for p in parts if p))
        return out

    return Dataset(inter, items, user_ids, texts, eda)
