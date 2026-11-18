"""The ancestor reconstruction pipeline as it stood at the end of 26 September 2026.

Two-descendant case (`reconstruct_pairwise`):
  1. Read an ancestor off the TCBB alignment of D2 against D1 and of D1 against D2,
     using the SIBLING scoring: duplication delta_0 = 10 (paper), ancestor-side
     deletion 10 per copy (the same as a duplication), descendant insertion 120 per
     copy (paper), mismatch 20 per differing nucleotide, match 0. Disputed columns
     take the variants-tree LCA.
  2. Keep the reading with the smaller d(A,D1) + d(A,D2) under the PAPER scoring
     (anchored).
  3. Polish by majority vote from the anchored alignments of D1 and D2 against the
     ancestor under the paper scoring, repeated until stable.

Many descendants (`reconstruct_multi`): the candidates are the pairwise readings of
all pairs; the one with the smallest total anchored paper cost to all descendants
is kept and then polished by the vote over all descendants.

Only the two sibling alignments in step 1 use the sibling scoring. Every distance
and the vote use the paper's penalties, as ruled on 26 September 2026.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .align import align
from .ancestor import position_votes, reconstruct_by_alignment
from .scoring import GAP, Scoring, hamming

MOTIF_LENGTH_DEFAULT = 6


def sibling_scoring(variants: Dict[str, str], mismatch: float = 20.0,
                    deletion_per_copy: float = 10.0, insertion_per_copy: float = 120.0,
                    motif_length: Optional[int] = None,
                    distance: Callable[[str, str], int] = hamming) -> Scoring:
    """Scoring for the two sibling alignments (see module docstring).

    `distance` is the variant distance (Hamming by default; pass
    tr_aligner.levenshtein.levenshtein for variants of unequal length).
    """
    L = motif_length or (len(next(iter(variants.values()))) if variants else MOTIF_LENGTH_DEFAULT)
    return Scoring(variants=dict(variants), match_reward=0.0, mismatch_penalty=-mismatch,
                   gap_penalty=-insertion_per_copy / L, ancestor_gap_penalty=-deletion_per_copy / L,
                   distance=distance)


def paper_scoring(variants: Dict[str, str]) -> Scoring:
    """Section 4.2 of the paper: match +50|a|, mismatch -50 H, gap -20|a|."""
    return Scoring(variants=dict(variants))


def total_cost(A: Sequence[str], Ds: Sequence[Sequence[str]], sc: Scoring) -> float:
    return sum(align(A, D, sc, anchored=True).distance for D in Ds)


def vote(A: Sequence[str], Ds: Sequence[Sequence[str]], sc: Scoring, max_rounds: int = 10
         ) -> Tuple[List[str], int]:
    """Majority-vote polish over all descendants (paper scoring, anchored)."""
    cur = list(A)
    for r in range(max_rounds):
        alns = [align(cur, D, sc, anchored=True) for D in Ds]
        votes = position_votes(cur, alns)
        new: List[str] = []
        for j, v in enumerate(votes):
            if not v:
                new.append(cur[j]); continue
            top = max(v.values())
            winners = [s for s, c in v.items() if c == top]
            choice = cur[j] if (cur[j] in winners or len(winners) > 1) else winners[0]
            if choice != GAP:
                new.append(choice)
        if new == cur:
            return cur, r
        cur = new
    return cur, max_rounds


@dataclass
class Result:
    ancestor: List[str]
    initial: List[str]
    candidates: List[Tuple[str, List[str], float]] = field(default_factory=list)  # (label, ancestor, cost)
    vote_rounds: int = 0


def reconstruct_pairwise(D1: Sequence[str], D2: Sequence[str], variants: Dict[str, str],
                         sib: Optional[Scoring] = None, paper: Optional[Scoring] = None,
                         polish: bool = True,
                         distance: Callable[[str, str], int] = hamming) -> Result:
    sib = sib or sibling_scoring(variants, distance=distance)
    paper = paper or paper_scoring(variants)
    ra = reconstruct_by_alignment(D1, D2, variants, scoring=sib, distance=distance)
    cands = []
    for label in ("12", "21"):
        a = ra[label].ancestor
        cands.append((label, a, total_cost(a, [D1, D2], paper)))
    best = min(cands, key=lambda c: c[2])
    initial = list(best[1])
    res = Result(ancestor=initial, initial=initial, candidates=cands)
    if polish:
        res.ancestor, res.vote_rounds = vote(initial, [D1, D2], paper)
    return res


def reconstruct_multi(Ds: Sequence[Sequence[str]], variants: Dict[str, str],
                      sib: Optional[Scoring] = None, paper: Optional[Scoring] = None,
                      polish: bool = True,
                      distance: Callable[[str, str], int] = hamming) -> Result:
    if len(Ds) < 2:
        raise ValueError("need at least two descendants")
    sib = sib or sibling_scoring(variants, distance=distance)
    paper = paper or paper_scoring(variants)
    cands = []
    for i, j in itertools.combinations(range(len(Ds)), 2):
        ra = reconstruct_by_alignment(Ds[i], Ds[j], variants, scoring=sib, distance=distance)
        for label in ("12", "21"):
            a = ra[label].ancestor
            cands.append((f"{i}{j}/{label}", a, total_cost(a, Ds, paper)))
    best = min(cands, key=lambda c: c[2])
    initial = list(best[1])
    res = Result(ancestor=initial, initial=initial, candidates=cands)
    if polish:
        res.ancestor, res.vote_rounds = vote(initial, Ds, paper)
    return res
