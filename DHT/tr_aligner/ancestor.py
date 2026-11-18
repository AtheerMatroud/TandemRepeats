"""Ancestor reconstruction for two tandem repeat maps (Step 2 of the study).

Given TR maps D1 and D2 of two homologous tandem repeats, estimate their most
recent common ancestor A. This is Problem 1 of the TCBB paper, left open there,
and follows thesis Chapter 7, Section 7.4 (approximate LCS with lowest common
ancestor substitution) with the variants graph, variant frequency and spread of
thesis Chapter 3, Sections 3.3 to 3.6, and the re-alignment correction
suggested in the thesis conclusion (Section 7.6).

Pipeline
--------
1. Pool the variants of D1 and D2; record frequency and spread of each variant.
2. Build the variants distance graph and a minimum spanning tree (Kruskal).
3. Root the tree at the ancestral variant: most frequent, ties broken by the
   larger spread, then by lexicographic order for determinism.
4. Approximate LCS of D1 and D2: two symbols may be paired when their variant
   distance is at most `threshold` (thesis: 2). Among LCSs of maximal length,
   the one with the smallest total distance across paired columns is chosen, so
   identical symbols are preferred to near matches.
5. Each paired column (x, y) contributes LCA_root(x, y) to the ancestor;
   unpaired symbols are dropped (they are taken to be duplications or
   insertions in one descendant only).
6. Optional refinement: align D1 and D2 against the ancestor with the TCBB
   algorithm; at each ancestral position collect the descendant symbols placed
   there (one vote per visit) and replace the symbol by the majority, the
   current symbol winning ties. Repeat until stable or `max_rounds`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .align import Alignment, align
from .scoring import GAP, Scoring, hamming


# ---------------------------------------------------------------------------
# 1. frequency and spread
# ---------------------------------------------------------------------------

@dataclass
class VariantStats:
    frequency: Dict[str, int]
    spread: Dict[str, int]            # rightmost - leftmost position, pooled over both sequences

    @classmethod
    def from_sequences(cls, seqs: Sequence[Sequence[str]]) -> "VariantStats":
        freq: Counter = Counter()
        first: Dict[str, int] = {}
        last: Dict[str, int] = {}
        offset = 0
        for s in seqs:
            for pos, sym in enumerate(s):
                freq[sym] += 1
                p = offset + pos
                first.setdefault(sym, p)
                last[sym] = p
            offset += len(s)
        spread = {v: last[v] - first[v] for v in freq}
        return cls(dict(freq), spread)

    def root(self) -> str:
        return max(self.frequency, key=lambda v: (self.frequency[v], self.spread[v], -ord(v[0]), v))


# ---------------------------------------------------------------------------
# 2 and 3. variants graph, minimum spanning tree, rooted LCA
# ---------------------------------------------------------------------------

@dataclass
class VariantTree:
    symbols: List[str]
    dist: Dict[str, Dict[str, int]]
    edges: List[Tuple[str, str, int]]              # MST edges (u, v, weight)
    root: str
    parent: Dict[str, Optional[str]] = field(default_factory=dict)
    depth: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(cls, variants: Dict[str, str], stats: VariantStats,
              distance: Callable[[str, str], int] = hamming,
              root: Optional[str] = None) -> "VariantTree":
        syms = sorted(variants)
        dist = {u: {v: (0 if u == v else distance(variants[u], variants[v])) for v in syms} for u in syms}
        # Kruskal on the complete distance graph; ties broken by symbol order for determinism
        cand = sorted(((dist[u][v], u, v) for i, u in enumerate(syms) for v in syms[i + 1:]))
        comp = {s: s for s in syms}

        def find(x: str) -> str:
            while comp[x] != x:
                comp[x] = comp[comp[x]]
                x = comp[x]
            return x

        edges: List[Tuple[str, str, int]] = []
        for w, u, v in cand:
            ru, rv = find(u), find(v)
            if ru != rv:
                comp[ru] = rv
                edges.append((u, v, w))
        r = root if root is not None else stats.root()
        tree = cls(syms, dist, edges, r)
        tree._root_at(r)
        return tree

    def _root_at(self, root: str) -> None:
        adj: Dict[str, List[str]] = defaultdict(list)
        for u, v, _ in self.edges:
            adj[u].append(v); adj[v].append(u)
        self.root = root
        self.parent = {root: None}
        self.depth = {root: 0}
        stack = [root]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in self.parent:
                    self.parent[v] = u
                    self.depth[v] = self.depth[u] + 1
                    stack.append(v)

    def lca(self, u: str, v: str) -> str:
        while self.depth[u] > self.depth[v]:
            u = self.parent[u]
        while self.depth[v] > self.depth[u]:
            v = self.parent[v]
        while u != v:
            u, v = self.parent[u], self.parent[v]
        return u

    def path_to_root(self, u: str) -> List[str]:
        out = [u]
        while self.parent[out[-1]] is not None:
            out.append(self.parent[out[-1]])
        return out

    def describe(self) -> str:
        lines = [f"root: {self.root}"]
        for u, v, w in self.edges:
            lines.append(f"  {u} -- {v}  (distance {w})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. approximate longest common subsequence
# ---------------------------------------------------------------------------

Column = Tuple[Optional[str], Optional[str]]


def approximate_lcs(D1: Sequence[str], D2: Sequence[str],
                    dist: Dict[str, Dict[str, int]], threshold: int = 2) -> List[Column]:
    """Columns of an approximate LCS alignment of D1 and D2.

    A column (x, y) pairs D1 symbol x with D2 symbol y when dist[x][y] <= threshold.
    Unpaired symbols appear as (x, None) or (None, y). Maximises the number of
    paired columns, then minimises the summed distance over paired columns.
    """
    n, m = len(D1), len(D2)
    NEG = (-1, 0)
    # L[i][j] = (pairs, -distance_sum) best for prefixes D1[:i], D2[:j]
    L = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    P = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best, move = L[i - 1][j], "u"
            if L[i][j - 1] > best:
                best, move = L[i][j - 1], "l"
            d = dist[D1[i - 1]][D2[j - 1]]
            if d <= threshold:
                c = (L[i - 1][j - 1][0] + 1, L[i - 1][j - 1][1] - d)
                if c > best:
                    best, move = c, "d"
            L[i][j] = best
            P[i][j] = move
    cols: List[Column] = []
    i, j = n, m
    while i > 0 or j > 0:
        mv = P[i][j] if i > 0 and j > 0 else ("u" if i > 0 else "l")
        if mv == "d":
            cols.append((D1[i - 1], D2[j - 1])); i -= 1; j -= 1
        elif mv == "u":
            cols.append((D1[i - 1], None)); i -= 1
        else:
            cols.append((None, D2[j - 1])); j -= 1
    cols.reverse()
    return cols


# ---------------------------------------------------------------------------
# 5 and 6. ancestor construction and refinement
# ---------------------------------------------------------------------------

@dataclass
class Reconstruction:
    ancestor: List[str]               # final estimate
    initial: List[str]                # before refinement
    columns: List[Column]             # approximate LCS columns
    tree: VariantTree
    stats: VariantStats
    rounds: int                       # refinement rounds applied
    history: List[List[str]] = field(default_factory=list)   # ancestor after each round
    votes: List[Dict[str, int]] = field(default_factory=list)  # last round's votes per position
    arbitration_log: List[Tuple[int, str, str, str, str]] = field(default_factory=list)  # (pos, x, y, lca, chosen)
    pairing_alignment: Optional[Alignment] = field(default=None, repr=False)
    trimmed: Optional[tuple] = field(default=None, repr=False)   # (T1, T2, dropped D1 idx, dropped D2 idx)

    def trace(self) -> str:
        lines = [self.tree.describe(), "approximate LCS columns (D1 / D2 -> ancestor):"]
        for x, y in self.columns:
            if x is None or y is None:
                lines.append(f"  {x or '-'} / {y or '-'}  -> dropped")
            else:
                a = x if x == y else self.tree.lca(x, y)
                tag = "" if x == y else f"  (LCA of {x},{y})"
                lines.append(f"  {x} / {y}  -> {a}{tag}")
        lines.append(f"initial ancestor : {''.join(self.initial)}")
        for r, h in enumerate(self.history, start=1):
            lines.append(f"after round {r}    : {''.join(h)}")
        lines.append(f"final ancestor   : {''.join(self.ancestor)}")
        return "\n".join(lines)


def position_votes(A: Sequence[str], alignments: Sequence[Alignment]) -> List[Dict[str, int]]:
    """For each ancestral position, count the descendant symbols placed against it.

    Every visit of a_j in the walk contributes one vote: the matched or
    substituted descendant symbol, or GAP when a_j is aligned against a gap.
    Descendant symbols inserted against a gap in the walk cast no vote.
    """
    votes: List[Dict[str, int]] = [defaultdict(int) for _ in A]
    for aln in alignments:
        for e in aln.events:
            if e.kind in ("M", "S"):
                votes[e.start - 1][aln.descendant[e.desc - 1]] += 1
            elif e.kind == "D":
                votes[e.start - 1][GAP] += 1
    return [dict(v) for v in votes]


def refine_once(A: Sequence[str], D1: Sequence[str], D2: Sequence[str],
                scoring: Scoring, **align_kw) -> Tuple[List[str], List[Dict[str, int]]]:
    """One majority-rule correction of A from its alignments with D1 and D2."""
    A = list(A)
    alns = [align(A, D1, scoring, **align_kw), align(A, D2, scoring, **align_kw)]
    votes = position_votes(A, alns)
    new: List[str] = []
    for j, v in enumerate(votes):
        cur = A[j]
        if not v:
            new.append(cur); continue
        top = max(v.values())
        winners = [s for s, c in v.items() if c == top]
        if cur in winners or len(winners) > 1:
            choice = cur                         # ties keep the current symbol
        else:
            choice = winners[0]
        if choice != GAP:
            new.append(choice)
    return new, votes


def pair_cost(A: Sequence[str], D1: Sequence[str], D2: Sequence[str],
              scoring: Scoring, **align_kw) -> float:
    """d(A, D1) + d(A, D2): the paper's criterion for judging a candidate ancestor."""
    return align(A, D1, scoring, **align_kw).distance + align(A, D2, scoring, **align_kw).distance


def arbitrate_by_cost(initial: Sequence[str], cols: Sequence[Column], D1: Sequence[str],
                      D2: Sequence[str], scoring: Scoring, tree: VariantTree,
                      **align_kw) -> Tuple[List[str], List[Tuple[int, str, str, str, str]]]:
    """Resolve each disputed LCS column (x != y) by cost instead of by tree LCA.

    Left to right, for each disputed column try the D1 symbol x and the D2 symbol y
    (and the LCA already in place) and keep the candidate ancestor with the lowest
    d(A, D1) + d(A, D2); ties keep the LCA. All candidates have the same length, so
    the criterion's bias towards length does not enter. Returns the ancestor and a
    record (position, x, y, lca, chosen) per disputed column.
    """
    A = list(initial)
    paired = [(x, y) for x, y in cols if x is not None and y is not None]
    log: List[Tuple[int, str, str, str, str]] = []
    for pos, (x, y) in enumerate(paired):
        if x == y:
            continue
        lca = A[pos]
        best_sym, best_cost = lca, pair_cost(A, D1, D2, scoring, **align_kw)
        for cand in dict.fromkeys([x, y]):
            if cand == lca:
                continue
            trial = A[:pos] + [cand] + A[pos + 1:]
            c = pair_cost(trial, D1, D2, scoring, **align_kw)
            if c < best_cost:
                best_sym, best_cost = cand, c
        A[pos] = best_sym
        log.append((pos + 1, x, y, lca, best_sym))
    return A, log


def reconstruct(D1: Sequence[str], D2: Sequence[str], variants: Dict[str, str],
                threshold: int = 2, refine: bool = False, max_rounds: int = 5,
                scoring: Optional[Scoring] = None,
                distance: Callable[[str, str], int] = hamming,
                root: Optional[str] = None, arbitration: str = "lca",
                **align_kw) -> Reconstruction:
    """Estimate the common ancestor of TR maps D1 and D2 (see module docstring).

    arbitration: "lca" writes the tree LCA in disputed columns (thesis Section 7.4);
                 "cost" then re-decides each disputed column by d(A,D1)+d(A,D2)
                 between the two observed symbols and the LCA (arbitrate_by_cost).
    """
    D1, D2 = list(D1), list(D2)
    stats = VariantStats.from_sequences([D1, D2])
    used = {s: variants[s] for s in stats.frequency}
    tree = VariantTree.build(used, stats, distance=distance, root=root)
    cols = approximate_lcs(D1, D2, tree.dist, threshold)
    initial = [x if x == y else tree.lca(x, y) for x, y in cols if x is not None and y is not None]
    rec = Reconstruction(ancestor=list(initial), initial=initial, columns=cols,
                         tree=tree, stats=stats, rounds=0)
    if arbitration == "cost":
        sc = scoring or Scoring(variants=variants, distance=distance)
        arb, log = arbitrate_by_cost(initial, cols, D1, D2, sc, tree, **align_kw)
        rec.arbitration_log = log
        rec.ancestor = list(arb)
        initial = arb
    if refine:
        sc = scoring or Scoring(variants=variants, distance=distance)
        cur = list(initial)
        for r in range(max_rounds):
            new, votes = refine_once(cur, D1, D2, sc, **align_kw)
            rec.votes = votes
            rec.rounds = r + 1
            rec.history.append(list(new))
            if new == cur:
                break
            cur = new
        rec.ancestor = cur
    return rec


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------

def symbol_edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Plain unit-cost edit distance between two TR maps (substitution, indel of a symbol)."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i]
        for j, y in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def compare(true_A: Sequence[str], est_A: Sequence[str], scoring: Scoring) -> Dict[str, float]:
    """Distances between the true and the estimated ancestor."""
    return {
        "exact": float(list(true_A) == list(est_A)),
        "symbol_edit": symbol_edit_distance(true_A, est_A),
        "tcbb_true_to_est": align(true_A, est_A, scoring).distance,
        "tcbb_est_to_true": align(est_A, true_A, scoring).distance,
        "tcbb_true_to_est_anchored": align(true_A, est_A, scoring, anchored=True).distance,
    }


# ---------------------------------------------------------------------------
# Direction A: duplication-aware construction from the TCBB alignment of D1 against D2
# ---------------------------------------------------------------------------

def columns_from_alignment(X: Sequence[str], Y: Sequence[str], scoring: Scoring,
                           **align_kw) -> Tuple[List[Column], Alignment]:
    """Pair the symbols of X and Y using the asymmetric alignment of Y against X.

    X plays the ancestor, Y the descendant. From the walk:
      * an X position reached by a match or substitution is a backbone position and
        gives the column (X[j], Y[i]); when X[j] is visited more than once (Y duplicated
        it) the matching visit is preferred, otherwise the first;
      * an X position aligned against a gap (D event) is a copy only X has: excluded;
      * a Y symbol against a gap (I event) is a copy only Y has: excluded;
      * visits reached through a backward edge are Y's duplicated copies and are only
        used when no other visit of that position exists.
    Columns are returned in X order.
    """
    aln = align(X, Y, scoring, anchored=True, **align_kw)
    visits: Dict[int, List[Tuple[str, str]]] = defaultdict(list)   # j -> [(kind, y symbol)]
    after_backedge = False
    for e in aln.events:
        if e.kind == "R":
            after_backedge = True
            continue
        if e.kind in ("M", "S"):
            tag = "copy" if after_backedge else "main"
            visits[e.start].append((e.kind, Y[e.desc - 1], tag))
        elif e.kind == "D":
            visits[e.start].append(("D", None, "main"))
        after_backedge = False
    cols: List[Column] = []
    for j in range(1, len(X) + 1):
        vs = visits.get(j, [])
        pairs = [v for v in vs if v[0] in ("M", "S")]
        if not pairs:
            continue                      # deleted: X-only copy
        main = [v for v in pairs if v[2] == "main"] or pairs
        match = [v for v in main if v[0] == "M"] or [v for v in pairs if v[0] == "M"]
        chosen = match[0] if match else main[0]
        cols.append((X[j - 1], chosen[1]))
    return cols, aln


def reconstruct_by_alignment(D1: Sequence[str], D2: Sequence[str], variants: Dict[str, str],
                             refine: bool = False, max_rounds: int = 5,
                             scoring: Optional[Scoring] = None,
                             distance: Callable[[str, str], int] = hamming,
                             root: Optional[str] = None,
                             direction: str = "both") -> Dict[str, Reconstruction]:
    """Direction A: ancestor from the duplication-aware pairing of D1 and D2.

    direction: "12" (D1 as ancestor, D2 as descendant), "21", or "both" (returns both).
    Disputed columns take the tree LCA; the optional anchored majority-vote polish is
    applied when refine=True. Returns {direction: Reconstruction}.
    """
    D1, D2 = list(D1), list(D2)
    sc = scoring or Scoring(variants=variants, distance=distance)
    stats = VariantStats.from_sequences([D1, D2])
    used = {s: variants[s] for s in stats.frequency}
    tree = VariantTree.build(used, stats, distance=distance, root=root)
    out: Dict[str, Reconstruction] = {}
    for d in (["12", "21"] if direction == "both" else [direction]):
        if d == "12":
            cols, aln = columns_from_alignment(D1, D2, sc)
            initial = [x if x == y else tree.lca(x, y) for x, y in cols]
        else:
            cols, aln = columns_from_alignment(D2, D1, sc)
            cols = [(y, x) for x, y in cols]      # keep (D1 symbol, D2 symbol) order
            initial = [x if x == y else tree.lca(x, y) for x, y in cols]
        rec = Reconstruction(ancestor=list(initial), initial=initial, columns=cols,
                             tree=tree, stats=stats, rounds=0)
        rec.pairing_alignment = aln
        if refine:
            cur = list(initial)
            for r in range(max_rounds):
                new, votes = refine_once(cur, D1, D2, sc, anchored=True)
                rec.votes = votes; rec.rounds = r + 1; rec.history.append(list(new))
                if new == cur:
                    break
                cur = new
            rec.ancestor = cur
        out[d] = rec
    return out


def descendant_only_indices(X: Sequence[str], Y: Sequence[str], scoring: Scoring) -> set:
    """Indices (0-based) of Y that the alignment of Y against X explains as Y-only material:
    symbols reached through a backward edge (duplicated copies) or inserted against a gap."""
    aln = align(X, Y, scoring, anchored=True)
    drop = set()
    after_backedge = False
    for e in aln.events:
        if e.kind == "R":
            after_backedge = True
            continue
        if e.kind == "I":
            drop.add(e.desc - 1)
        elif e.kind in ("M", "S") and after_backedge:
            drop.add(e.desc - 1)
        after_backedge = False
    return drop


def reconstruct_symmetric(D1: Sequence[str], D2: Sequence[str], variants: Dict[str, str],
                          refine: bool = False, max_rounds: int = 5,
                          scoring: Optional[Scoring] = None,
                          distance: Callable[[str, str], int] = hamming,
                          root: Optional[str] = None, threshold: int = 2) -> Reconstruction:
    """Direction A, symmetric form.

    1. Align D2 against D1: the copies of D2 reached by backward edges (and inserted
       symbols) are D2-only material; remove them.
    2. Align D1 against D2: likewise remove D1-only material.
    3. Pair the two trimmed sequences: column by column when their lengths agree,
       otherwise by the approximate LCS; disputed columns take the tree LCA.
    4. Optional anchored majority-vote polish.
    """
    D1, D2 = list(D1), list(D2)
    sc = scoring or Scoring(variants=variants, distance=distance)
    stats = VariantStats.from_sequences([D1, D2])
    used = {s: variants[s] for s in stats.frequency}
    tree = VariantTree.build(used, stats, distance=distance, root=root)
    drop2 = descendant_only_indices(D1, D2, sc)
    drop1 = descendant_only_indices(D2, D1, sc)
    T1 = [s for i, s in enumerate(D1) if i not in drop1]
    T2 = [s for i, s in enumerate(D2) if i not in drop2]
    if len(T1) == len(T2):
        cols: List[Column] = list(zip(T1, T2))
    else:
        cols = approximate_lcs(T1, T2, tree.dist, threshold)
    initial = [x if x == y else tree.lca(x, y) for x, y in cols if x is not None and y is not None]
    rec = Reconstruction(ancestor=list(initial), initial=initial, columns=cols,
                         tree=tree, stats=stats, rounds=0)
    rec.trimmed = (T1, T2, sorted(drop1), sorted(drop2))
    if refine:
        cur = list(initial)
        for r in range(max_rounds):
            new, votes = refine_once(cur, D1, D2, sc, anchored=True)
            rec.votes = votes; rec.rounds = r + 1; rec.history.append(list(new))
            if new == cur:
                break
            cur = new
        rec.ancestor = cur
    return rec
