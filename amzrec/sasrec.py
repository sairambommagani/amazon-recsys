"""SASRec: self-attentive sequential recommendation (Kang & McAuley, 2018), plus a text-aware variant.

A small transformer reads a user's purchases in time order (the last 50) and, at every position, predicts
the next product. Only the past is visible (causal attention), exactly as when recommending live.

Product vectors:
  sasrec        a learned ID embedding per product
  sasrec_text   ID embedding + a learned projection of the product's MiniLM text vector. During training
                the ID part of ~20% of products is randomly switched off ("ID dropout"), so the model
                learns to rely on the text too. A brand-new product has no trained ID embedding, so it is
                represented by its text alone, and can be scored for every user like any other product.

Training target: the next purchase, but the loss only counts purchases the user rated 4-5 stars, so the
model learns what people buy AND like (1-3 star purchases still appear in the input sequence as context).
Loss: sampled softmax with 256 random negative products per batch (full softmax over all products would
be too slow on a CPU).
"""
from __future__ import annotations

import copy
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def sequences(inter: pd.DataFrame, users: np.ndarray, cutoff: int, L: int, with_days: bool = False):
    """Last L products (item+1, 0 = padding, left-padded) and their 'liked' flags, per user, before cutoff
    (plus the purchase day of each position if with_days). `inter` must be sorted by user, then day."""
    sub = inter[(inter["day"] < cutoff) & inter["u"].isin(users)]
    sub = sub.groupby("u", sort=False).tail(L)
    seq = np.zeros((len(users), L), dtype=np.int64)
    pos = np.zeros((len(users), L), dtype=np.int8)
    row = pd.Index(users).get_indexer(sub["u"].to_numpy())
    rank = sub.groupby("u", sort=False).cumcount(ascending=False).to_numpy()     # 0 = most recent
    col = L - 1 - rank
    seq[row, col] = sub["i"].to_numpy() + 1
    pos[row, col] = sub["pos"].to_numpy()
    if with_days:
        days = np.zeros((len(users), L), dtype=np.int32)
        days[row, col] = sub["day"].to_numpy()
        return seq, pos, days
    return seq, pos


def _trim(*t: torch.Tensor):
    """Drop leading columns that are padding for every row of the batch (sequences are left-padded)."""
    nz = (t[0] > 0).any(0).nonzero()
    start = int(nz[0]) if len(nz) else t[0].shape[1] - 1
    out = tuple(x[:, start:] for x in t)
    return out if len(out) > 1 else out[0]


class Block(nn.Module):
    """Pre-norm transformer block: masked multi-head self-attention + feed-forward, both with residuals."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.p = heads, dropout
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(B, L, d)))
        return x + self.drop(self.ff(self.n2(x)))


class SASRec(nn.Module):
    def __init__(self, n_items: int, d: int = 64, L: int = 50, blocks: int = 2, heads: int = 2, dropout: float = 0.2,
                 text: np.ndarray | None = None, id_dropout: float = 0.2):
        super().__init__()
        self.n_items, self.d, self.L, self.heads = n_items, d, L, heads
        self.item = nn.Embedding(n_items + 1, d, padding_idx=0)
        nn.init.normal_(self.item.weight, std=0.02)
        with torch.no_grad():
            self.item.weight[0].zero_()
        self.use_text = text is not None
        self.id_dropout = id_dropout
        if self.use_text:
            T = np.vstack([np.zeros((1, text.shape[1]), np.float32), text.astype(np.float32)])
            self.register_buffer("text", torch.from_numpy(T))
            self.proj = nn.Linear(text.shape[1], d, bias=False)
        self.register_buffer("seen", torch.ones(n_items + 1, dtype=torch.bool))  # products with a trained ID
        self.pos = nn.Embedding(L, d)
        self.blocks = nn.ModuleList([Block(d, heads, dropout) for _ in range(blocks)])
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def item_vectors(self, train: bool = False) -> torch.Tensor:
        """Vectors of all products (row 0 = padding)."""
        e = self.item.weight
        if self.use_text:
            keep = self.seen.float()
            if train and self.id_dropout > 0:
                keep = keep * (torch.rand(e.shape[0], device=e.device) > self.id_dropout).float()
            e = e * keep[:, None] + self.proj(self.text)
        return e

    def encode(self, seq: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """seq is left-padded and may be shorter than L (padding columns trimmed); positions count from
        the most recent purchase, so the last column always gets position L-1."""
        B, L = seq.shape
        x = V[seq] * math.sqrt(self.d) + self.pos.weight[None, self.L - L:]
        x = self.drop(x) * (seq > 0).unsqueeze(-1)
        causal = torch.ones(L, L, dtype=torch.bool, device=seq.device).tril()
        allowed = causal[None] & (seq > 0)[:, None, :]
        allowed |= torch.eye(L, dtype=torch.bool, device=seq.device)[None]       # padding rows see themselves
        allowed = allowed.unsqueeze(1)                                              # (B, 1, L, L), True = may attend
        for blk in self.blocks:
            x = blk(x, allowed)
        return self.norm(x)

    @torch.no_grad()
    def user_vectors(self, seq: np.ndarray, batch: int = 1024) -> np.ndarray:
        """Representation after the last purchase = what the user wants next."""
        self.eval()
        V = self.item_vectors()
        out = []
        for s in range(0, len(seq), batch):
            h = self.encode(_trim(torch.from_numpy(seq[s:s + batch])), V)
            out.append(h[:, -1].numpy())
        return np.vstack(out) if out else np.zeros((0, self.d), np.float32)

    @torch.no_grad()
    def product_vectors(self) -> np.ndarray:
        self.eval()
        return self.item_vectors()[1:].numpy()


def train_sasrec(model: SASRec, seq: np.ndarray, pos: np.ndarray, epochs: int = 30, batch: int = 256, lr: float = 2e-3,
                 negatives: int = 256, eval_fn=None, patience: int = 3, log=print, seed: int = 0,
                 weight: np.ndarray | None = None) -> tuple[SASRec, int, list]:
    """seq/pos: (users, L+1). Inputs are positions 0..L-1, targets 1..L. eval_fn(model) -> validation score;
    training stops after `patience` epochs without improvement and the best epoch's weights are kept.
    weight: optional (users, L+1) loss weight per target (time decay: recent purchases count more)."""
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    X, Y, W = torch.from_numpy(seq[:, :-1]), torch.from_numpy(seq[:, 1:]), torch.from_numpy(pos[:, 1:].astype(bool))
    G = torch.from_numpy((np.ones(seq.shape, np.float32) if weight is None else weight.astype(np.float32))[:, 1:])
    keep = (W & (Y > 0)).any(1).numpy()                                   # users with at least one liked target
    X, Y, W, G = X[keep], Y[keep], W[keep], G[keep]
    length = (X > 0).sum(1).numpy()
    history, best, best_ep, best_state, bad = [], -1.0, 0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        # batches of users with similar sequence length -> little padding (much faster on CPU)
        order = np.lexsort((rng.random(len(X)), length))
        batches = [order[s:s + batch] for s in range(0, len(order), batch)]
        for bi in rng.permutation(len(batches)):
            b = torch.from_numpy(batches[bi])
            x, y, w, g = _trim(X[b], Y[b], W[b], G[b])
            w = w & (y > 0)
            V = model.item_vectors(train=True)
            h = model.encode(x, V)
            neg = torch.randint(1, model.n_items + 1, (negatives,))
            pos_logit = (h * V[y]).sum(-1, keepdim=True)
            neg_logit = h @ V[neg].T
            neg_logit = neg_logit.masked_fill(y.unsqueeze(-1) == neg[None, None, :], -1e9)   # a negative = the target
            logits = torch.cat([pos_logit, neg_logit], -1)[w]
            per = F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long), reduction="none")
            loss = (per * g[w]).sum() / g[w].sum().clamp_min(1e-8)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()); nb += 1
        msg = f"    epoch {ep:2d}: loss {tot / max(nb, 1):.4f} ({time.time() - t0:.0f}s)"
        if eval_fn is not None:
            score = eval_fn(model)
            history.append(round(float(score), 5))
            msg += f", validation NDCG@10 {score:.4f}"
            if score > best:
                best, best_ep, best_state, bad = score, ep, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
        log(msg)
        if eval_fn is not None and bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, (best_ep if eval_fn is not None else epochs), history
