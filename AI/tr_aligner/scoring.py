"""Scoring matrix sigma and duplication cost vector delta (TCBB paper, Sections 3.1 and 4.2).

Default parameters follow Section 4.2 of the paper:

    sigma(alpha, alpha) = +50 |alpha|
    sigma(alpha, beta)  = -50 H(alpha, beta)      alpha != beta
    sigma(alpha, -) = sigma(-, alpha) = -20 |alpha|
    delta_i = d + i * eps,  i = 0, 1, ...,  with d = -10, eps = -1

so that a 1-duplication costs delta_0 = -10 and an l-duplication costs delta_{l-1}.
(Ruling of 26 Sep 2026: delta is indexed from 0, so delta_0 = -10.)

The distance between two variants defaults to the Hamming distance, which is
what the paper uses. For variants of unequal length use
tr_aligner.levenshtein.levenshtein_scoring instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Sequence

GAP = "-"


def hamming(a: str, b: str) -> int:
    """Hamming distance between two strings of equal length."""
    if len(a) != len(b):
        raise ValueError(
            f"Hamming distance needs equal lengths ({len(a)} vs {len(b)}); "
            "use tr_aligner.levenshtein for variants of unequal length"
        )
    return sum(x != y for x, y in zip(a, b))


def delta_vector(n: int, d: float = -10.0, eps: float = -1.0) -> list[float]:
    """delta_i = d + i*eps for i = 0..n-1; delta_{l-1} is the cost of an l-duplication.

    Conditions (D1) and (D2) of the paper hold whenever d <= 0 and eps <= 0.
    """
    if d > 0 or eps > 0:
        raise ValueError("d and eps must be non-positive for (D1) and (D2) to hold")
    return [d + i * eps for i in range(n)]


@dataclass
class Scoring:
    """Holds sigma for a given variant table.

    variants  : mapping from variant symbol (e.g. 'a') to its nucleotide string.
                Symbol lengths |alpha| are taken from the nucleotide strings.
                If a symbol is missing, |alpha| falls back to `default_length`.
    distance  : function on two nucleotide strings (default Hamming).
    """

    variants: Mapping[str, str] = field(default_factory=dict)
    match_reward: float = 50.0
    mismatch_penalty: float = -50.0
    gap_penalty: float = -20.0
    ancestor_gap_penalty: Optional[float] = None   # per nucleotide, for an ancestor symbol against a gap; None = gap_penalty
    distance: Callable[[str, str], int] = hamming
    default_length: int = 1
    _cache: Dict[tuple, float] = field(default_factory=dict, repr=False)

    def length(self, symbol: str) -> int:
        seq = self.variants.get(symbol)
        return len(seq) if seq is not None else self.default_length

    def sigma(self, a: str, d: str) -> float:
        """sigma(a, d) with a from the ancestor walk and d from the descendant."""
        key = (a, d)
        if key in self._cache:
            return self._cache[key]
        if a == GAP and d == GAP:
            val = float("-inf")
        elif a == GAP:
            val = self.gap_penalty * self.length(d)
        elif d == GAP:
            g = self.gap_penalty if self.ancestor_gap_penalty is None else self.ancestor_gap_penalty
            val = g * self.length(a)
        elif a == d:
            val = self.match_reward * self.length(a)
        else:
            sa, sd = self.variants.get(a), self.variants.get(d)
            dist = self.distance(sa, sd) if sa is not None and sd is not None else 1
            val = self.mismatch_penalty * dist
        self._cache[key] = val
        return val

    def matrix(self, symbols: Sequence[str]) -> Dict[str, Dict[str, float]]:
        """Explicit sigma table over the given symbols plus the gap, for inspection."""
        syms = list(symbols) + [GAP]
        return {x: {y: self.sigma(x, y) for y in syms} for x in syms}
