"""Tests on small synthetic files with the real Amazon Reviews 2023 JSON format."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    out = tmp_path_factory.mktemp("art")
    subprocess.run([sys.executable, "-m", "amzrec.train", "--synthetic", "--fast", "--encoder", "tfidf", "--out", str(out)],
                   cwd=ROOT, check=True)
    return out


def test_k_core():
    from amzrec.data import k_core
    # users 0-2 bought products 0-2; user 3 bought only product 0 and product 3 was bought once
    df = pd.DataFrame({"u": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 0], "i": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 3]})
    out = k_core(df, 3, log=lambda *a: None)
    assert set(out["u"]) == {0, 1, 2} and set(out["i"]) == {0, 1, 2}


def test_sequences_left_padded_most_recent_last():
    from amzrec.sasrec import sequences
    inter = pd.DataFrame({"u": [0, 0, 0, 1], "i": [5, 6, 7, 9], "day": [1, 2, 3, 1], "pos": [1, 0, 1, 1]})
    seq, pos = sequences(inter, np.array([0, 1]), cutoff=3, L=4)          # day 3 is excluded
    assert seq[0].tolist() == [0, 0, 6, 7] and pos[0].tolist() == [0, 0, 1, 0]
    assert seq[1].tolist() == [0, 0, 0, 10]


def test_sasrec_trims_padding_consistently():
    import torch
    from amzrec.sasrec import SASRec
    m = SASRec(20, d=16, L=6)
    a = np.array([[0, 0, 0, 3, 4, 5]])
    b = np.array([[3, 4, 5]])
    m.eval()
    with torch.no_grad():
        V = m.item_vectors()
        ha = m.encode(torch.from_numpy(a), V)[:, -1]
        hb = m.encode(torch.from_numpy(b), V)[:, -1]
    assert torch.allclose(ha, hb, atol=1e-5)                                  # padding does not change the result


def test_pipeline_results(artifacts):
    m = json.loads((artifacts / "metrics.json").read_text())
    old, new = m["old_products"]["results"], m["new_products"]["results"]
    assert old["als"]["ndcg@10"] > old["popularity"]["ndcg@10"]
    assert old["fusion"]["ndcg@10"] >= 0.9 * max(v["ndcg@10"] for k, v in old.items() if k != "fusion")
    assert new["text_to_factors"]["ndcg@10"] > new["random"]["ndcg@10"]
    assert "5_core" in m["eda"] and m["eda"]["all_data"]["reviews"] > 0
    for f in ("items.parquet", "E.npy", "V_als.npy", "knn.npz", "sasrec.pt", "sasrec_text.pt", "history.parquet", "config.json"):
        assert (artifacts / f).exists(), f


def test_api(artifacts, monkeypatch):
    import amzrec.api as api
    monkeypatch.setattr(api, "ART", artifacts)
    api._cache.clear()
    from fastapi.testclient import TestClient
    c = TestClient(api.app)
    users = c.get("/users/sample?n=2").json()
    r = c.get(f"/users/{users[0]['user_id']}/recommendations").json()
    assert len(r["recommendations"]) == 10 and all(x["reasons"] for x in r["recommendations"])
    bought = {x["item_id"] for x in r["recent_purchases"]}
    assert not bought & {x["item_id"] for x in r["recommendations"]}           # never recommends what was bought
    iid = r["recommendations"][0]["item_id"]
    sim = c.get(f"/items/{iid}/similar?k=3").json()
    assert len(sim["similar_description"]) == 3 and len(sim["bought_next"]) == 3
    nx = c.post("/next-items", json={"item_ids": [r["recent_purchases"][0]["item_id"]]}).json()
    assert len(nx["recommendations"]) == 10
    res = c.get("/search", params={"q": "zombie survival", "k": 5}).json()["results"]
    assert sum("Horror" in x["title"] for x in res) >= 4
    assert c.get("/users/unknown/recommendations").status_code == 404
    assert c.post("/next-items", json={"item_ids": ["nope"]}).status_code == 404
