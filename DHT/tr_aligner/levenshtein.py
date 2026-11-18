"""Levenshtein distance and a Scoring for variants of unequal length.

Kept separate from scoring.py on purpose: the paper's simulations use the
Hamming distance because every simulated variant has the motif length. Real
data (thesis Chapter 3, taro variant 'v' of length 13 among 11-mers; TCBB
Example 1, variants c and e) contain single nucleotide insertions and
deletions inside a motif copy, so the variant distance must allow indels.
Use `levenshtein_scoring(variants)` in that setting.
"""

from __future__ import annotations

from typing import Mapping

from .scoring import Scoring


def levenshtein(a: str, b: str) -> int:
    """Unit-cost edit distance (substitution, insertion, deletion) between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1,            # deletion
                           cur[j - 1] + 1,         # insertion
                           prev[j - 1] + (ca != cb)))  # substitution
        prev = cur
    return prev[-1]


def levenshtein_scoring(variants: Mapping[str, str], **kwargs) -> Scoring:
    """Scoring identical to the paper's, except that mismatches are charged
    mismatch_penalty * Levenshtein(alpha, beta) so variants of different length
    can be compared. Gap and match terms still scale with |alpha|.
    """
    return Scoring(variants=dict(variants), distance=levenshtein, **kwargs)
