"""Reconstruction of the ancestor of (D1, D2) by a trained model."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import fastalign as fa
from .data import (BOS, EOS, OFF, PAD, SEP, UNK, Seq2SeqDataset, TagDataset, TokenBatchSampler,
                   collate_seq2seq, collate_tag)
from .models import build


def to_device(b, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.no_grad()
def greedy_decode(model, b, extra: int = 20):
    """Token-by-token decoding; stops at EOS or at max(n(D1), n(D2)) + extra symbols."""
    B = b["src"].size(0)
    n1 = (b["seg"] == 0).logical_and(b["src"] != PAD).sum(1) - 1
    n2 = (b["seg"] == 1).logical_and(b["src"] != PAD).sum(1)
    max_len = int(torch.maximum(n1, n2).max()) + extra
    mem = model.encode(b)
    ys = torch.full((B, 1), BOS, dtype=torch.long, device=b["src"].device)
    done = torch.zeros(B, dtype=torch.bool, device=ys.device)
    for _ in range(max_len):
        nxt = model.decode(b, mem, ys)[:, -1].argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
        ys = torch.cat([ys, nxt.unsqueeze(1)], 1)
        done |= nxt == EOS
        if bool(done.all()):
            break
    out = []
    for row in ys[:, 1:].tolist():
        seq = []
        for tok in row:
            if tok in (EOS, PAD):
                break
            if tok >= OFF:
                seq.append(tok - OFF)
        out.append(seq)
    return out


@torch.no_grad()
def predict(model, families: List[dict], device, max_tokens: int = 40000, workers: int = 2,
            progress: bool = False) -> List[List[int]]:
    """The reconstructed ancestor (list of variant ids) of every family, in order."""
    model.eval()
    amp = device.type == "cuda"
    out: List[List[int]] = [None] * len(families)
    if model.kind == "tagger":
        ds = TagDataset(families, augment=False, with_labels=False)
        dl = DataLoader(ds, batch_sampler=TokenBatchSampler(ds.lengths, max_tokens, False), collate_fn=collate_tag,
                        num_workers=workers)
        for b in dl:
            bd = to_device(b, device)
            with torch.autocast(device.type, enabled=amp):
                lab = model(bd).argmax(-1).cpu().numpy()
            for row, i in zip(lab, b["idx"].tolist()):
                c = ds.cols[i]
                out[i] = fa.apply_labels(c["cx"], c["cy"], row[:len(c["cx"])])
    else:
        ds = Seq2SeqDataset(families, augment=False)
        dl = DataLoader(ds, batch_sampler=TokenBatchSampler(ds.lengths, max_tokens, False),
                        collate_fn=collate_seq2seq, num_workers=workers)
        for k, b in enumerate(dl):
            bd = to_device(b, device)
            with torch.autocast(device.type, enabled=amp):
                seqs = greedy_decode(model, bd)
            for seq, i in zip(seqs, b["idx"].tolist()):
                out[i] = seq
            if progress and (k + 1) % 10 == 0:
                print(f"  decoded {k + 1}/{len(dl)} batches", flush=True)
    return out


def load_model(name: str, ckpt_path: str, device):
    ck = torch.load(ckpt_path, map_location=device)
    model = build(name, **ck["cfg"]["model"]).to(device)
    model.load_state_dict(ck["model"])
    return model, ck
