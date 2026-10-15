"""Text encoders: item text (plot, genres, keywords / categories, attributes, reviews) -> unit vectors.

  minilm  sentence-transformers/all-MiniLM-L6-v2, a small (22M parameters) sentence-embedding model.
          It understands meaning, so "a robbery that goes wrong" finds heist films that never use the
          word "robbery". Runs on CPU; the model (~90 MB) is downloaded once from Hugging Face.
  tfidf   TF-IDF (words + word pairs) compressed to 256 dimensions with truncated SVD (latent semantic
          analysis). Works offline, no extra packages; matches words rather than meaning.
  auto    minilm if sentence-transformers is installed and the model can be loaded, else tfidf.
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np

MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def _unit(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


class TfidfEncoder:
    kind = "tfidf"

    def __init__(self, dim: int = 256, max_features: int = 200_000):
        self.dim, self.max_features = dim, max_features

    def fit_encode(self, texts: list[str], log=print) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(min_df=2, max_df=0.5, max_features=self.max_features, ngram_range=(1, 2),
                                   sublinear_tf=True, stop_words="english", dtype=np.float32)
        try:
            M = self.vec.fit_transform(texts)
        except ValueError:                                  # tiny / repetitive corpus: keep every word
            self.vec.set_params(min_df=1, max_df=1.0)
            M = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(max(1, min(self.dim, M.shape[1] - 1, M.shape[0] - 1)), random_state=0)
        return _unit(self.svd.fit_transform(M))

    def encode(self, texts: list[str]) -> np.ndarray:
        return _unit(self.svd.transform(self.vec.transform(texts)))

    def save(self, path: Path):
        joblib.dump(self, path)


class MiniLMEncoder:
    kind = "minilm"

    def __init__(self, name: str = MINILM):
        from sentence_transformers import SentenceTransformer
        self.name = name
        self.model = SentenceTransformer(name, device="cpu")
        self.model.max_seq_length = 128                    # first ~100 words; 2x faster than 256 on CPU

    def fit_encode(self, texts: list[str], log=print) -> np.ndarray:
        out, bs, t0 = [], 4096, time.time()
        for s in range(0, len(texts), bs):
            out.append(self.model.encode(texts[s:s + bs], batch_size=64, normalize_embeddings=True,
                                         convert_to_numpy=True))
            done = min(s + bs, len(texts))
            log(f"    encoded {done:,}/{len(texts):,} texts ({time.time() - t0:.0f}s)")
        return _unit(np.vstack(out))

    def encode(self, texts: list[str]) -> np.ndarray:
        return _unit(self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True))

    def save(self, path: Path):
        joblib.dump({"kind": "minilm", "name": self.name}, path)


def make_encoder(kind: str = "auto", log=print):
    if kind in ("auto", "minilm"):
        try:
            return MiniLMEncoder()
        except Exception as e:                                           # not installed / no internet
            if kind == "minilm":
                raise
            log(f"  MiniLM not available ({type(e).__name__}: {str(e)[:120]}); using TF-IDF + SVD instead")
    return TfidfEncoder()


def load_encoder(path: Path):
    obj = joblib.load(path)
    if isinstance(obj, dict) and obj.get("kind") == "minilm":
        return MiniLMEncoder(obj["name"])
    return obj
