"""Tensors, augmentation and batching for the three candidate models.

Token ids: 0 PAD, 1 BOS, 2 EOS, 3 SEP (also the gap of an alignment column), 4 UNK,
5 + v for variant v of the family. A variant carries no meaning across families, so
every token is accompanied by the one-hot nucleotides of its variant (4 x 11 = 44
values), and variants are numbered in order of first appearance in D1 then D2.
A variant of A that occurs in neither descendant cannot be produced and is UNK.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from . import fastalign as fa

PAD, BOS, EOS, SEP, UNK, OFF = 0, 1, 2, 3, 4, 5
GAP = SEP
VOCAB = 128
MAXV = VOCAB - OFF
NT = {"A": 0, "C": 1, "G": 2, "T": 3}
MOTIF = 11
N_REP = 5                      # look-behind / look-ahead range of the repeat features


def onehot_variants(variants: Sequence[str]) -> np.ndarray:
    out = np.zeros((VOCAB, 4 * MOTIF), dtype=np.float32)
    for i, v in enumerate(variants[:MAXV]):
        for p, c in enumerate(v[:MOTIF]):
            out[OFF + i, 4 * p + NT[c]] = 1.0
    return out


def canonical(fam: dict, swap: bool = False, reverse: bool = False) -> dict:
    """The family with D1/D2 optionally swapped and all sequences optionally reversed,
    renumbered in order of first appearance in D1 then D2 (A-only variants last)."""
    D1, D2, A = (fam["D2"], fam["D1"], fam["A"]) if swap else (fam["D1"], fam["D2"], fam["A"])
    if reverse:
        D1, D2, A = D1[::-1], D2[::-1], A[::-1]
    if not swap and not reverse:
        return fam
    new: Dict[int, int] = {}
    for seq in (D1, D2):
        for v in seq:
            if v not in new:
                new[v] = len(new)
    nD = len(new)
    for v in A:
        if v not in new:
            new[v] = len(new)
    variants = [None] * len(new)
    for old, n in new.items():
        variants[n] = fam["variants"][old]
    out = dict(fam)
    out.update(variants=variants, nD=nD, A=[new[v] for v in A], D1=[new[v] for v in D1], D2=[new[v] for v in D2])
    return out


def _tok(seq, nD):
    return [OFF + v if v < min(nD, MAXV) else UNK for v in seq]


# ---------------------------------------------------------------------------
# sequence-to-sequence models (Transformer, BiLSTM)
# ---------------------------------------------------------------------------

class Seq2SeqDataset(Dataset):
    def __init__(self, families: List[dict], augment: bool = False):
        self.fams, self.augment = families, augment
        self.lengths = [len(f["D1"]) + len(f["D2"]) + len(f["A"]) for f in families]

    def __len__(self):
        return len(self.fams)

    def __getitem__(self, i):
        f = self.fams[i]
        if self.augment:
            f = canonical(f, swap=random.random() < 0.5, reverse=random.random() < 0.5)
        nD = f["nD"]
        d1, d2 = _tok(f["D1"], nD), _tok(f["D2"], nD)
        src = d1 + [SEP] + d2
        seg = [0] * (len(d1) + 1) + [1] * len(d2)
        pos = list(range(len(d1) + 1)) + list(range(len(d2)))
        tgt = [BOS] + _tok(f["A"], nD) + [EOS]
        vmask = np.zeros(VOCAB, dtype=bool)
        vmask[[EOS, UNK]] = True
        vmask[OFF:OFF + min(nD, MAXV)] = True
        return {"src": src, "seg": seg, "pos": pos, "tgt": tgt, "vtab": onehot_variants(f["variants"]),
                "vmask": vmask, "idx": i}


def _pad(seqs, value=0, dtype=torch.long):
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), L), value, dtype=dtype)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.as_tensor(s, dtype=dtype)
    return out


def collate_seq2seq(items):
    return {"src": _pad([x["src"] for x in items]), "seg": _pad([x["seg"] for x in items]),
            "pos": _pad([x["pos"] for x in items]), "tgt": _pad([x["tgt"] for x in items]),
            "vtab": torch.from_numpy(np.stack([x["vtab"] for x in items])),
            "vmask": torch.from_numpy(np.stack([x["vmask"] for x in items])),
            "idx": torch.tensor([x["idx"] for x in items])}


# ---------------------------------------------------------------------------
# alignment-guided tagger
# ---------------------------------------------------------------------------

def _rep_features(seq: np.ndarray) -> np.ndarray:
    """For every position p and l = 1..N_REP: seq[p] == seq[p-l] and seq[p] == seq[p+l]."""
    n = len(seq)
    out = np.zeros((n, 2 * N_REP), dtype=np.float32)
    for l in range(1, N_REP + 1):
        if n > l:
            eq = (seq[l:] == seq[:-l]).astype(np.float32)
            out[l:, l - 1] = eq
            out[:-l, N_REP + l - 1] = eq
    return out


_COLS: Dict[str, dict] = {}


def columns(fam: dict, with_labels: bool = True) -> dict:
    """Alignment columns of D1 and D2 with their features (and labels, if A is used).
    Computed once per family and kept in memory."""
    out = _COLS.get(fam["id"])
    if out is None:
        H = fa.hamming_matrix(fam["variants"])
        d1, d2 = np.asarray(fam["D1"], dtype=np.int32), np.asarray(fam["D2"], dtype=np.int32)
        cx, cy = fa.nw_align(d1, d2, H)
        n = len(cx)
        px, py = np.cumsum(cx >= 0) - 1, np.cumsum(cy >= 0) - 1
        r1, r2 = _rep_features(d1), _rep_features(d2)
        rep = np.zeros((n, 4 * N_REP), dtype=np.float32)
        mx, my = cx >= 0, cy >= 0
        rep[mx, :2 * N_REP] = r1[px[mx]]
        rep[my, 2 * N_REP:] = r2[py[my]]
        ham = np.full(n, MOTIF + 1, dtype=np.int64)
        both = mx & my
        ham[both] = H[cx[both], cy[both]]
        out = _COLS[fam["id"]] = {"cx": cx, "cy": cy, "ham": ham, "rep": rep}
    if with_labels and "lab" not in out:
        lab, k = fa.column_labels(out["cx"], out["cy"], fam["A"])
        out["lab"] = lab.astype(np.int64)
        out["expressible"] = int(k)
    return out


class TagDataset(Dataset):
    """Columns are computed once; the D1/D2 swap is applied on the fly."""

    def __init__(self, families: List[dict], augment: bool = False, with_labels: bool = True):
        self.fams, self.augment, self.with_labels = families, augment, with_labels
        self.cols = [columns(f, with_labels) for f in families]
        self.vtabs = [onehot_variants(f["variants"]) for f in families]
        self.lengths = [len(c["cx"]) for c in self.cols]

    def __len__(self):
        return len(self.fams)

    def __getitem__(self, i):
        c = self.cols[i]
        cx, cy, rep = c["cx"], c["cy"], c["rep"]
        lab = c.get("lab") if self.with_labels else None
        if self.augment and random.random() < 0.5:
            cx, cy = cy, cx
            rep = np.concatenate([rep[:, 2 * N_REP:], rep[:, :2 * N_REP]], axis=1)
            if lab is not None:
                lab = np.where(lab == 1, 2, np.where(lab == 2, 1, 0))
        if lab is not None:      # canonical label for identical symbols is KEEP_X
            lab = np.where((lab != 0) & (cx == cy), 1, lab)
        x = np.where(cx >= 0, np.minimum(cx, MAXV - 1) + OFF, GAP)
        y = np.where(cy >= 0, np.minimum(cy, MAXV - 1) + OFF, GAP)
        return {"x": x, "y": y, "ham": c["ham"], "rep": rep, "lab": lab, "vtab": self.vtabs[i], "idx": i}


def collate_tag(items):
    L = max(len(x["x"]) for x in items)
    B = len(items)
    out = {"x": torch.zeros(B, L, dtype=torch.long), "y": torch.zeros(B, L, dtype=torch.long),
           "ham": torch.zeros(B, L, dtype=torch.long), "rep": torch.zeros(B, L, 4 * N_REP),
           "lab": torch.full((B, L), -100, dtype=torch.long),
           "vtab": torch.from_numpy(np.stack([x["vtab"] for x in items])),
           "idx": torch.tensor([x["idx"] for x in items])}
    for i, it in enumerate(items):
        n = len(it["x"])
        out["x"][i, :n] = torch.from_numpy(it["x"].astype(np.int64))
        out["y"][i, :n] = torch.from_numpy(it["y"].astype(np.int64))
        out["ham"][i, :n] = torch.from_numpy(it["ham"])
        out["rep"][i, :n] = torch.from_numpy(it["rep"])
        if it["lab"] is not None:
            out["lab"][i, :n] = torch.from_numpy(it["lab"].astype(np.int64))
    return out


# ---------------------------------------------------------------------------
# batching by length under a token budget
# ---------------------------------------------------------------------------

class TokenBatchSampler(Sampler):
    def __init__(self, lengths: Sequence[int], max_tokens: int, shuffle: bool, max_batch: int = 64):
        self.lengths, self.max_tokens, self.shuffle, self.max_batch = lengths, max_tokens, shuffle, max_batch
        self.batches = self._build()

    def _build(self):
        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        batches, cur, cur_max = [], [], 0
        for i in order:
            m = max(cur_max, self.lengths[i])
            if cur and (m * (len(cur) + 1) > self.max_tokens or len(cur) >= self.max_batch):
                batches.append(cur); cur, m = [], self.lengths[i]
            cur.append(i); cur_max = m
        if cur:
            batches.append(cur)
        return batches

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
