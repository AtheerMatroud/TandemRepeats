"""Simulation of tandem repeat maps: ancestors, descendants and event histories.

Events follow Section 5 of the TCBB paper: deletion : substitution : duplication
at rate 1 : 3 : 10, duplicated or deleted blocks of one to three motif copies,
substitutions to a variant at Hamming distance 1 when one exists. Everything is
driven by an explicit random.Random so datasets are reproducible.

Default variants are the six of thesis Chapter 1.3 (Figure 1.1).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .scoring import hamming

THESIS_VARIANTS: Dict[str, str] = {
    "a": "CATGGT", "b": "TATGGA", "c": "CACGCT",
    "d": "TATGGT", "e": "CAAGGT", "f": "CACGGT",
}


@dataclass
class Simulator:
    variants: Dict[str, str]
    rates: Tuple[float, float, float] = (1.0, 3.0, 10.0)   # deletion, substitution, duplication
    max_block: int = 3
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.symbols = list(self.variants)

    def random_ancestor(self, length: int, weights: Optional[Sequence[float]] = None) -> str:
        if weights is None:
            return "".join(self.rng.choice(self.symbols) for _ in range(length))
        return "".join(self.rng.choices(self.symbols, weights=list(weights), k=length))

    def one_event(self, seq: Sequence[str]) -> Tuple[List[str], str]:
        """Apply one event; returns (new sequence, event label in walk notation)."""
        seq = list(seq)
        n = len(seq)
        kind = self.rng.choices(["del", "sub", "dup"], weights=list(self.rates))[0]
        if kind == "dup":
            L = self.rng.randint(1, min(self.max_block, n))
            s = self.rng.randint(0, n - L)
            seq[s + L:s + L] = seq[s:s + L]
            return seq, f"R{s + 1}-{s + L}"
        if kind == "del" and n > 3:
            L = self.rng.randint(1, min(self.max_block, n - 2))
            s = self.rng.randint(0, n - L)
            del seq[s:s + L]
            return seq, f"D{s + 1}-{s + L}"
        p = self.rng.randrange(n)
        cur = seq[p]
        near = [v for v in self.symbols if v != cur
                and len(self.variants[v]) == len(self.variants[cur])
                and hamming(self.variants[v], self.variants[cur]) == 1]
        seq[p] = self.rng.choice(near or [v for v in self.symbols if v != cur])
        return seq, f"S{p + 1}:{cur}->{seq[p]}"

    def descend(self, A: Sequence[str], k: int) -> Tuple[str, List[str]]:
        """k events applied to A; returns (descendant, event labels)."""
        seq = list(A)
        labels: List[str] = []
        for _ in range(k):
            seq, e = self.one_event(seq)
            labels.append(e)
        return "".join(seq), labels

    def family(self, length: int, n_desc: int, k: int) -> Tuple[str, List[str], List[List[str]]]:
        """One ancestor and n_desc descendants, each k events away."""
        A = self.random_ancestor(length)
        Ds, hist = [], []
        for _ in range(n_desc):
            D, h = self.descend(A, k)
            Ds.append(D); hist.append(h)
        return A, Ds, hist

    def dataset(self, n: int, length: int, n_desc: int, k: int):
        return [self.family(length, n_desc, k) for _ in range(n)]


def dup_then_sub(A: str, dup_start: int, dup_len: int, sub_pos: int, sub_to: str) -> Tuple[str, str]:
    """Deterministic toy protocol of Step 2: one duplication (1-based start, length)
    followed by one substitution (1-based position in the duplicated sequence)."""
    A = list(A)
    D = A[:dup_start - 1 + dup_len] + A[dup_start - 1:dup_start - 1 + dup_len] + A[dup_start - 1 + dup_len:]
    old = D[sub_pos - 1]
    D[sub_pos - 1] = sub_to
    return "".join(D), f"R{dup_start}-{dup_start + dup_len - 1} then {old}->{sub_to} at {sub_pos}"


def history_weight(labels: Sequence[str], variants: Dict[str, str], motif_length: Optional[int] = None,
                   duplication: str = "paper", d: float = 10.0, eps: float = 1.0) -> float:
    """Weight of a true event history (the paper's d_hist, Section 5).

    duplication="paper" (default, ruling of 27 September 2026): the paper's formula
        1 + 10(l-1) for an l-copy duplication (1, 11, 21, ...).
    duplication="delta": d + (l-1) eps, the delta vector the alignment uses (10, 11, 12, ...);
        the two agree only at l = 2.
    Deletion of l copies: 20 l |X|; substitution: 50 per differing nucleotide.
    """
    L = motif_length or len(next(iter(variants.values())))
    w = 0.0
    for lab in labels:
        if lab.startswith("R"):
            s, e = (int(x) for x in lab[1:].split("-")); l = e - s + 1
            w += (1 + 10 * (l - 1)) if duplication == "paper" else (d + (l - 1) * eps)
        elif lab.startswith("D"):
            s, e = (int(x) for x in lab[1:].split("-")); w += 20 * (e - s + 1) * L
        elif lab.startswith("S"):
            a, b = lab.split(":")[1].split("->"); w += 50 * hamming(variants[a], variants[b])
    return w


# ---------------------------------------------------------------------------
# TCBB Section 5 protocol (rulings of 27 September 2026)
# ---------------------------------------------------------------------------

def _inverse_length(rng: random.Random, max_len: int) -> int:
    """l in 1..max_len with probability proportional to 1/l."""
    weights = [1.0 / l for l in range(1, max_len + 1)]
    return rng.choices(range(1, max_len + 1), weights=weights)[0]


@dataclass
class TCBBSimulator:
    """Generator following Section 5.1 of the TCBB paper and the Java TR_GenMain,
    with the rulings of 27 September 2026:

    * motif: random 11-mer; ancestor starts as 50 s identical copies;
    * burn-in: events at deletion : substitution : duplication = 0.2 : 1 : 4 while the
      length is at or below the target, 4 : 1 : 0.2 above it; duplication and deletion
      blocks of 1..5 copies with probability proportional to 1/l; stops when the
      sequence holds `n_variants` (15) distinct motif variants (restart after `guard`
      events if not reached);
    * descendant: 5 t events at 0.2 : 1 : 4; duplications 1..5 and deletions 1..3 copies,
      both by 1/l; substitutions change one random nucleotide to another, creating new
      variants freely; block start uniform where the block fits;
    * history weight (d_hist): duplication of l copies 9 + l (the alignment's delta),
      deletion 20 |X| per copy, substitution 50 per nucleotide.
    Sequences are lists of nucleotide strings (motif copies); `to_symbols` maps a set of
    sequences to TR maps over a shared variant table.
    """
    motif_length: int = 11
    n_variants: int = 15
    rates: Tuple[float, float, float] = (0.2, 1.0, 4.0)     # deletion, substitution, duplication
    dup_max: int = 5
    del_max_burnin: int = 5
    del_max_desc: int = 3
    guard: int = 100_000
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    # -- primitives -------------------------------------------------------
    def random_motif(self) -> str:
        return "".join(self.rng.choice("ACGT") for _ in range(self.motif_length))

    def _substitute(self, copy: str) -> str:
        i = self.rng.randrange(len(copy))
        new = self.rng.choice([c for c in "ACGT" if c != copy[i]])
        return copy[:i] + new + copy[i + 1:]

    def _event(self, seq: List[str], rates, dup_max: int, del_max: int) -> Tuple[List[str], str]:
        kind = self.rng.choices(["del", "sub", "dup"], weights=list(rates))[0]
        n = len(seq)
        if kind == "dup":
            l = min(_inverse_length(self.rng, dup_max), n)
            s = self.rng.randint(0, n - l)
            seq[s + l:s + l] = seq[s:s + l]
            return seq, f"R{s + 1}-{s + l}"
        if kind == "del" and n > 3:
            l = min(_inverse_length(self.rng, del_max), n - 2)
            s = self.rng.randint(0, n - l)
            del seq[s:s + l]
            return seq, f"D{s + 1}-{s + l}"
        p = self.rng.randrange(n)
        old = seq[p]
        seq[p] = self._substitute(old)
        return seq, f"S{p + 1}"

    # -- ancestor and descendants -----------------------------------------
    def ancestor(self, s: int) -> List[str]:
        target = 50 * s
        while True:
            motif = self.random_motif()
            seq = [motif] * target
            for _ in range(self.guard):
                rates = self.rates if len(seq) <= target else (self.rates[2], self.rates[1], self.rates[0])
                seq, _ = self._event(seq, rates, self.dup_max, self.del_max_burnin)
                if len(set(seq)) >= self.n_variants:
                    return seq

    def descend(self, A: Sequence[str], t: int) -> Tuple[List[str], List[str], float]:
        """5t events; returns (descendant, labels, d_hist)."""
        seq = list(A)
        labels: List[str] = []
        w = 0.0
        for _ in range(5 * t):
            n_before = len(seq)
            seq, lab = self._event(seq, self.rates, self.dup_max, self.del_max_desc)
            labels.append(lab)
            if lab.startswith("R"):
                a, b = (int(x) for x in lab[1:].split("-")); w += 9 + (b - a + 1)
            elif lab.startswith("D"):
                a, b = (int(x) for x in lab[1:].split("-")); w += 20 * self.motif_length * (b - a + 1)
            else:
                w += 50
        return seq, labels, w

    def family(self, s: int, t: int, m: int):
        A = self.ancestor(s)
        Ds, hist, weights = [], [], []
        for _ in range(m):
            D, h, w = self.descend(A, t)
            Ds.append(D); hist.append(h); weights.append(w)
        return A, Ds, hist, weights


def to_symbols(sequences: Sequence[Sequence[str]]):
    """Map motif-copy sequences to TR maps over a shared variant table.
    Returns (list of symbol strings, variants dict symbol -> nucleotide string)."""
    from .mapping import VariantMap
    vm = VariantMap.from_sequences(sequences)
    # TR maps are returned as LISTS of symbols: beyond 52 variants the symbols are
    # multi-character ('v0', 'v1', ...) and must never be joined into a string.
    return [vm.map(list(s)) for s in sequences], dict(vm.variant_of)
