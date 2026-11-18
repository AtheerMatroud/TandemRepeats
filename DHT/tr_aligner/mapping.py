"""Tandem repeat mapping (TCBB paper, Section 2.2).

Two tasks live here:

1. VariantMap: assign a symbol to every distinct motif variant seen in one or
   more sequences of motif copies, and translate between copies and symbols
   (the TR map tau_v(S)). Symbols are assigned in order of first appearance,
   a..z then A..Z then v0, v1, ... so that hand examples stay readable.

2. parse_tandem_repeat: split a DNA string into approximate copies of a motif
   with the wrap-around dynamic programme of Fischetti et al. (1993), ported
   from the JavaScript `parseSequence`. This is step 3 of the mapping process
   and is only needed when starting from nucleotides rather than TR maps.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from .levenshtein import levenshtein
from .scoring import hamming


def _symbol_stream() -> Iterable[str]:
    yield from string.ascii_lowercase
    yield from string.ascii_uppercase
    k = 0
    while True:
        yield f"v{k}"
        k += 1


@dataclass
class VariantMap:
    """Bidirectional map between motif-copy strings and single symbols."""

    symbol_of: Dict[str, str] = field(default_factory=dict)
    variant_of: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_sequences(cls, sequences: Iterable[Sequence[str]]) -> "VariantMap":
        vm = cls()
        for seq in sequences:
            for copy in seq:
                vm.add(copy)
        return vm

    def add(self, copy: str) -> str:
        if copy in self.symbol_of:
            return self.symbol_of[copy]
        for sym in _symbol_stream():
            if sym not in self.variant_of:
                self.symbol_of[copy] = sym
                self.variant_of[sym] = copy
                return sym
        raise RuntimeError("unreachable")

    def map(self, copies: Sequence[str]) -> List[str]:
        """tau_v: motif copies -> symbols (adding unseen copies)."""
        return [self.add(c) for c in copies]

    def unmap(self, symbols: Sequence[str]) -> List[str]:
        return [self.variant_of[s] for s in symbols]

    def distance_matrix(self, use_levenshtein: bool = False) -> Dict[str, Dict[str, int]]:
        dist = levenshtein if use_levenshtein else hamming
        syms = list(self.variant_of)
        return {x: {y: (0 if x == y else dist(self.variant_of[x], self.variant_of[y]))
                    for y in syms} for x in syms}

    def all_equal_length(self) -> bool:
        lengths = {len(v) for v in self.variant_of.values()}
        return len(lengths) <= 1


def parse_tandem_repeat(
    sequence: str,
    motif: str,
    match: int = 2,
    mismatch: int = -1,
    gap: int = -2,
) -> List[str]:
    """Split `sequence` into approximate copies of `motif` (wrap-around DP).

    Returns the list of copies in order. A copy may contain '-' where a motif
    position was deleted, and may be longer than the motif where nucleotides
    were inserted. Copies at either end may be partial.
    Ported from tr-aligner/domain.js parseSequence; scoring defaults kept.
    """
    S, P = sequence, motif
    n, L = len(S), len(P)
    score = [[0] * (L + 1) for _ in range(n + 1)]
    ev = [[""] * (L + 1) for _ in range(n + 1)]
    for j in range(L + 1):
        ev[0][j] = "L"
    for i in range(1, n + 1):
        for j in range(1, L + 1):
            diag = score[i - 1][j - 1] + (match if S[i - 1] == P[j - 1] else mismatch)
            up = score[i - 1][j] + gap
            left = score[i][j - 1] + gap
            best = max(diag, up, left)
            score[i][j] = best
            if best == diag:
                ev[i][j] = "M" if S[i - 1] == P[j - 1] else "m"
            elif best == up:
                ev[i][j] = "U"
            else:
                ev[i][j] = "L"
        # wrap: column 0 of this row takes the value of the last column
        score[i][0] = score[i][L]
        ev[i][0] = "P"
        for j in range(1, L + 1):
            left = score[i][j - 1] + gap
            if left > score[i][j]:
                score[i][j] = left
                ev[i][j] = "L"
    # traceback
    i, j = n, L
    copies: List[str] = []
    buf = ""
    while i >= 1 or j >= 1:
        e = ev[i][j]
        if e in ("M", "m"):
            buf = S[i - 1] + buf; i -= 1; j -= 1
        elif e == "U":
            buf = S[i - 1] + buf; i -= 1
        elif e == "L":
            buf = "-" + buf; j -= 1
        elif e == "P":
            copies.append(buf); buf = ""; j = L
        else:
            break
    copies.append(buf)
    copies.reverse()
    return [c for c in copies if c.strip("-")]
