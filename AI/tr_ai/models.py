"""The three candidate models.

transformer  encoder-decoder Transformer: [D1 ; SEP ; D2] -> A, token by token
bilstm       BiLSTM encoder, LSTM decoder with attention: same task
aligntag     alignment-guided tagger: for every column of the alignment of D1 and D2,
             predict whether A keeps the symbol of D1, keeps the symbol of D2, or
             has no copy there
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PAD, VOCAB, MOTIF, N_REP

NEG = -1e4


class VariantEmbed(nn.Module):
    """Token embedding plus a projection of the nucleotides of the token's variant."""

    def __init__(self, d: int):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d, padding_idx=PAD)
        self.nt = nn.Linear(4 * MOTIF, d, bias=False)

    def forward(self, tokens, vtab):
        feat = torch.gather(vtab, 1, tokens.unsqueeze(-1).expand(-1, -1, vtab.size(-1)))
        return self.tok(tokens) + self.nt(feat)


class VariantOut(nn.Module):
    """Logits over the tokens; a variant is also scored through its nucleotides."""

    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, VOCAB)
        self.q = nn.Linear(d, d, bias=False)
        self.nt = nn.Linear(4 * MOTIF, d, bias=False)
        self.scale = 1.0 / math.sqrt(d)

    def forward(self, h, vtab, vmask):
        logits = self.lin(h) + self.scale * torch.einsum("bld,bvd->blv", self.q(h), self.nt(vtab))
        return logits.masked_fill(~vmask.unsqueeze(1), NEG)


def sinusoid(pos, d):
    div = torch.exp(torch.arange(0, d, 2, device=pos.device, dtype=torch.float32) * (-math.log(10000.0) / d))
    ang = pos.unsqueeze(-1).float() * div
    return torch.stack([ang.sin(), ang.cos()], dim=-1).flatten(-2)


class TransformerS2S(nn.Module):
    kind = "seq2seq"

    def __init__(self, d=256, heads=4, enc_layers=4, dec_layers=4, ff=1024, dropout=0.1):
        super().__init__()
        self.d = d
        self.emb = VariantEmbed(d)
        self.seg = nn.Embedding(2, d)
        self.tf = nn.Transformer(d, heads, enc_layers, dec_layers, ff, dropout, batch_first=True, norm_first=True)
        self.out = VariantOut(d)
        self.drop = nn.Dropout(dropout)

    def encode(self, b):
        x = self.emb(b["src"], b["vtab"]) + self.seg(b["seg"]) + sinusoid(b["pos"], self.d)
        return self.tf.encoder(self.drop(x), src_key_padding_mask=b["src"] == PAD)

    def decode(self, b, mem, tgt_in):
        T = tgt_in.size(1)
        y = self.emb(tgt_in, b["vtab"]) + sinusoid(torch.arange(T, device=tgt_in.device).unsqueeze(0), self.d)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_in.device), 1)
        h = self.tf.decoder(self.drop(y), mem, tgt_mask=causal, tgt_key_padding_mask=tgt_in == PAD,
                            memory_key_padding_mask=b["src"] == PAD)
        return self.out(h, b["vtab"], b["vmask"])

    def forward(self, b):
        return self.decode(b, self.encode(b), b["tgt"][:, :-1])


class BiLSTMS2S(nn.Module):
    kind = "seq2seq"

    def __init__(self, d=256, layers=2, dropout=0.1):
        super().__init__()
        self.d = d
        self.emb = VariantEmbed(d)
        self.seg = nn.Embedding(2, d)
        self.enc = nn.LSTM(d, d // 2, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.dec = nn.LSTM(d, d, layers, batch_first=True, dropout=dropout)
        self.q = nn.Linear(d, d, bias=False)
        self.comb = nn.Linear(2 * d, d)
        self.out = VariantOut(d)
        self.drop = nn.Dropout(dropout)

    def encode(self, b):
        x = self.drop(self.emb(b["src"], b["vtab"]) + self.seg(b["seg"]) + sinusoid(b["pos"], self.d))
        lens = (b["src"] != PAD).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        mem, _ = self.enc(packed)
        mem, _ = nn.utils.rnn.pad_packed_sequence(mem, batch_first=True, total_length=b["src"].size(1))
        return mem

    def decode(self, b, mem, tgt_in):
        h, _ = self.dec(self.drop(self.emb(tgt_in, b["vtab"])))
        att = torch.einsum("btd,bsd->bts", self.q(h), mem) / math.sqrt(self.d)
        att = att.masked_fill((b["src"] == PAD).unsqueeze(1), NEG).softmax(-1)
        ctx = torch.einsum("bts,bsd->btd", att, mem)
        return self.out(torch.tanh(self.comb(torch.cat([h, ctx], -1))), b["vtab"], b["vmask"])

    def forward(self, b):
        return self.decode(b, self.encode(b), b["tgt"][:, :-1])


class AlignTagger(nn.Module):
    kind = "tagger"

    def __init__(self, d=256, layers=3, dropout=0.1):
        super().__init__()
        self.emb = VariantEmbed(d)
        self.ham = nn.Embedding(MOTIF + 2, d)
        self.rep = nn.Linear(4 * N_REP, d)
        self.inp = nn.Linear(3 * d, d)
        self.rnn = nn.LSTM(d, d, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 3))
        self.drop = nn.Dropout(dropout)

    def forward(self, b):
        ex, ey = self.emb(b["x"], b["vtab"]), self.emb(b["y"], b["vtab"])
        z = torch.cat([ex, ey, self.ham(b["ham"]) + self.rep(b["rep"])], -1)
        z = self.drop(torch.relu(self.inp(z)))
        lens = (b["x"] != PAD).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(z, lens, batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=b["x"].size(1))
        logits = self.out(h)
        from .data import GAP
        xg, yg = b["x"] == GAP, b["y"] == GAP
        logits = logits.clone()
        logits[..., 1] = logits[..., 1].masked_fill(xg, NEG)                    # cannot keep a gap
        logits[..., 2] = logits[..., 2].masked_fill(yg | (b["x"] == b["y"]), NEG)
        return logits


MODELS = {"transformer": TransformerS2S, "bilstm": BiLSTMS2S, "aligntag": AlignTagger}

DEFAULTS = {
    "transformer": dict(model=dict(d=256, heads=4, enc_layers=4, dec_layers=4, ff=1024, dropout=0.1),
                        lr=3e-4, epochs=40, max_tokens=24000, warmup=1000),
    "bilstm": dict(model=dict(d=256, layers=2, dropout=0.1), lr=1e-3, epochs=40, max_tokens=40000, warmup=500),
    "aligntag": dict(model=dict(d=256, layers=3, dropout=0.1), lr=1e-3, epochs=30, max_tokens=40000, warmup=300),
}


def build(name: str, **kw):
    return MODELS[name](**kw)
