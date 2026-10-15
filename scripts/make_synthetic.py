"""Small fake Amazon Reviews 2023 files (Video Games) with the REAL JSON format, for offline tests.

Built-in structure the models can find:
  * users prefer one platform and a couple of genres (collaborative + content signal, also in the text)
  * games come in series (Part 1 -> Part 2 -> Part 3): after buying part n, users often buy part n+1,
    which is an order effect only a sequential model can use
  * products are released over time, so new products appear in the validation and test periods
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PLATFORMS = ["PlayStation 4", "Xbox One", "Nintendo Switch", "PC"]
GENRES = {"RPG": "open world quest dragon magic sword story", "Shooter": "gun battle squad multiplayer war fps",
          "Racing": "cars speed track drift championship", "Sports": "football soccer league team season",
          "Puzzle": "logic brain puzzles levels relaxing", "Horror": "zombie survival dark fear monsters"}


def write(out: Path, n_users: int = 1500, n_series: int = 120, days: int = 1400, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    genres = list(GENRES)
    items = []
    for s in range(n_series):
        g, p = genres[s % len(genres)], PLATFORMS[(s // len(genres)) % len(PLATFORMS)]
        start = int(rng.integers(-300, days - 200))
        for part in range(int(rng.integers(1, 4))):
            items.append({"asin": f"B{len(items):09d}", "series": s, "part": part, "genre": g, "platform": p,
                          "release": start + part * int(rng.integers(120, 300)), "q": rng.pareto(2.0) + 0.3})
    with open(out / "meta_Video_Games.jsonl", "w") as f:
        for it in items:
            words = GENRES[it["genre"]].split()
            f.write(json.dumps({
                "main_category": "Video Games", "title": f"{it['genre']} Saga {it['series']} Part {it['part'] + 1} - {it['platform']}",
                "average_rating": 4.2, "rating_number": 10, "features": [f"{' '.join(rng.choice(words, 5))} adventure"],
                "description": [f"A {it['genre'].lower()} game with {' '.join(rng.choice(words, 6))}."],
                "price": round(float(rng.uniform(10, 60)), 2), "images": [], "videos": [], "store": f"Studio {it['series'] % 15}",
                "categories": ["Video Games", it["platform"], "Games"],
                "details": {"Platform": it["platform"], "Genre": it["genre"], "Release date": f"day {it['release']}"},
                "parent_asin": it["asin"], "bought_together": None}) + "\n")
    rel = np.array([it["release"] for it in items])
    q = np.array([it["q"] for it in items])
    g_of = np.array([genres.index(it["genre"]) for it in items])
    p_of = np.array([PLATFORMS.index(it["platform"]) for it in items])
    nxt = {k: k + 1 for k in range(len(items) - 1) if items[k + 1]["series"] == items[k]["series"]}
    with open(out / "Video_Games.jsonl", "w") as f:
        for u in range(n_users):
            pp = rng.integers(len(PLATFORMS))
            gp = np.full(len(genres), 0.2); gp[rng.choice(len(genres), 2, replace=False)] = 3.0
            d = int(rng.integers(0, days - 100))
            owned, last = set(), None
            for _ in range(int(rng.integers(4, 25))):
                d += int(rng.exponential(40)) + 1
                if d >= days:
                    break
                if last in nxt and rel[nxt[last]] <= d and nxt[last] not in owned and rng.random() < 0.6:
                    k = nxt[last]                                                # buy the sequel
                else:
                    w = q * gp[g_of] * np.where(p_of == pp, 3.0, 0.3) * (rel <= d) * np.exp(-np.maximum(d - rel, 0) / 500)
                    w[list(owned)] = 0
                    if w.sum() == 0:
                        continue
                    k = int(rng.choice(len(items), p=w / w.sum()))
                owned.add(k); last = k
                good = gp[g_of[k]] > 1 and p_of[k] == pp
                rating = int(np.clip(round((4.4 if good else 2.8) + rng.normal(0, 0.8)), 1, 5))
                f.write(json.dumps({"rating": float(rating), "title": "Great game" if rating >= 4 else "Not for me",
                                    "text": f"{' '.join(rng.choice(GENRES[items[k]['genre']].split(), 8))}",
                                    "images": [], "asin": items[k]["asin"], "parent_asin": items[k]["asin"],
                                    "user_id": f"AE{u:026d}", "timestamp": (1_500_000_000 + d * 86400) * 1000 + int(rng.integers(0, 86_400_000)),
                                    "helpful_vote": 0, "verified_purchase": bool(rng.random() < 0.9)}) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    print("wrote", write(Path(ap.parse_args().out)))
