"""Reconstruction of the duplication history tree of k homologous TR maps.

Step 1  distance between two TR maps X and Y through their reconstructed ancestor:
            d(X, Y) = d_est(Ahat, X) + d_est(Ahat, Y),   Ahat = Ahat(X, Y),
        anchored edit-weight scoring (as for MSY1 in the MPE manuscript).
Step 2  the tree, by one of two rules:
          closest pair      : join the pair with the smallest d, replace the pair by its
                              reconstructed ancestor, recompute d from that ancestor, repeat;
          neighbour joining : topology from the leaf distance matrix alone, then the
                              ancestors are reconstructed from the leaves upwards.
        Both stop at the closest common ancestor of all sequences (the root); the root
        is not reduced further.
Step 3  each ancestor is aligned against each of its children (asymmetric alignment,
        anchored edit-weight scoring); the events R, S, D, I on the edge are read off.
Step 4  each motif copy of a child is traced to the copy of its parent it was aligned
        against; composing the edges traces every copy of every leaf to a copy of the root.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from tr_aligner.align import align, Alignment
from tr_aligner.pipeline import paper_scoring, reconstruct_pairwise, sibling_scoring
from tr_aligner.scoring import Scoring, hamming


def edit_weight_scoring(variants: Dict[str, str], distance=hamming) -> Scoring:
    """Costs of the TCBB paper, Section 4.2, with no reward for a match."""
    return Scoring(variants=dict(variants), match_reward=0.0, distance=distance)


@dataclass
class Node:
    name: str
    seq: List[str]
    children: List["Node"] = field(default_factory=list)
    # filled by annotate_edges, describing the edge from the parent to this node
    parent: Optional["Node"] = None
    alignment: Optional[Alignment] = None
    events: str = ""
    d_est: float = 0.0
    origin: List[Optional[int]] = field(default_factory=list)   # origin[i] = parent copy (1-based) of copy i+1

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def leaves(self) -> List["Node"]:
        return [self] if self.is_leaf else [x for c in self.children for x in c.leaves()]

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


def pair_join(X: Sequence[str], Y: Sequence[str], variants: Dict[str, str], distance=hamming
              ) -> Tuple[List[str], float, float]:
    """Reconstructed ancestor of X and Y, and d_est(Ahat, X), d_est(Ahat, Y)."""
    A = reconstruct_pairwise(X, Y, variants, sib=sibling_scoring(variants, distance=distance),
                             paper=paper_scoring(variants) if distance is hamming
                             else Scoring(variants=dict(variants), distance=distance),
                             distance=distance).ancestor
    ew = edit_weight_scoring(variants, distance)
    return list(A), align(A, X, ew, anchored=True).distance, align(A, Y, ew, anchored=True).distance


def distance_matrix(nodes: Sequence[Node], variants: Dict[str, str], distance=hamming):
    """All pairwise joins: {(i, j): (Ahat, d_i, d_j)} for i < j."""
    return {(i, j): pair_join(nodes[i].seq, nodes[j].seq, variants, distance)
            for i, j in itertools.combinations(range(len(nodes)), 2)}


def closest_pair_tree(leaves: Dict[str, Sequence[str]], variants: Dict[str, str], distance=hamming,
                      log: Optional[list] = None) -> Node:
    """Rule 1: join the closest pair, replace it by its reconstructed ancestor, repeat."""
    nodes = [Node(n, list(s)) for n, s in leaves.items()]
    step = 0
    while len(nodes) > 1:
        step += 1
        joins = distance_matrix(nodes, variants, distance)
        dist = {ij: v[1] + v[2] for ij, v in joins.items()}
        best = min(dist.values())
        ties = [ij for ij, d in dist.items() if d == best]
        i, j = ties[0]
        A = joins[(i, j)][0]
        new = Node(f"A{step}", A, [nodes[i], nodes[j]])
        if log is not None:
            log.append({"step": step, "names": [n.name for n in nodes],
                        "dist": {(nodes[a].name, nodes[b].name): d for (a, b), d in dist.items()},
                        "ancestors": {(nodes[a].name, nodes[b].name): v[0] for (a, b), v in joins.items()},
                        "joined": (nodes[i].name, nodes[j].name), "ties": len(ties), "new": new.name,
                        "ancestor": A})
        nodes = [n for k, n in enumerate(nodes) if k not in (i, j)] + [new]
    return nodes[0]


def neighbour_joining_tree(leaves: Dict[str, Sequence[str]], variants: Dict[str, str], distance=hamming,
                           log: Optional[list] = None) -> Node:
    """Rule 2: neighbour-joining topology from the leaf distances, rooted on the last
    edge joined; then ancestors reconstructed from the leaves upwards."""
    nodes = [Node(n, list(s)) for n, s in leaves.items()]
    joins = distance_matrix(nodes, variants, distance)
    D: Dict[FrozenSet[int], float] = {frozenset(ij): v[1] + v[2] for ij, v in joins.items()}
    if log is not None:
        log.append({"names": [n.name for n in nodes],
                    "dist": {(nodes[a].name, nodes[b].name): v[1] + v[2] for (a, b), v in joins.items()}})
    active: Dict[int, Node] = dict(enumerate(nodes))
    nxt, step = len(nodes), 0
    while len(active) > 2:
        step += 1
        n = len(active)
        r = {a: sum(D[frozenset((a, b))] for b in active if b != a) for a in active}
        Q = {(a, b): (n - 2) * D[frozenset((a, b))] - r[a] - r[b]
             for a, b in itertools.combinations(sorted(active), 2)}
        a, b = min(Q, key=Q.get)
        for c in active:
            if c not in (a, b):
                D[frozenset((nxt, c))] = 0.5 * (D[frozenset((a, c))] + D[frozenset((b, c))] - D[frozenset((a, b))])
        if log is not None:
            log.append({"step": step, "Q": {(active[x].name, active[y].name): q for (x, y), q in Q.items()},
                        "joined": (active[a].name, active[b].name)})
        new = Node(f"A{step}", [], [active[a], active[b]])
        del active[a], active[b]
        active[nxt] = new
        nxt += 1
    a, b = sorted(active)
    root = Node(f"A{step + 1}", [], [active[a], active[b]])
    # ancestors from the leaves upwards
    def fill(node: Node) -> None:
        if node.is_leaf:
            return
        for c in node.children:
            fill(c)
        node.seq = pair_join(node.children[0].seq, node.children[1].seq, variants, distance)[0]
    fill(root)
    return root


def annotate_edges(root: Node, variants: Dict[str, str], distance=hamming) -> Node:
    """Steps 3 and 4: align every ancestor against its children; store events and copy origins."""
    ew = edit_weight_scoring(variants, distance)
    for node in root.walk():
        for c in node.children:
            aln = align(node.seq, c.seq, ew, anchored=True)
            c.parent, c.alignment, c.events, c.d_est = node, aln, aln.event_string(), aln.distance + 0.0
            c.origin = [None] * len(c.seq)
            for e in aln.events:
                if e.kind in ("M", "S") and e.desc is not None:
                    c.origin[e.desc - 1] = e.start
    return root


def trace_to_root(node: Node) -> List[Optional[int]]:
    """For each motif copy of `node`, the copy of the root it descends from (1-based), or None."""
    cur: List[Optional[int]] = list(range(1, len(node.seq) + 1))
    while node.parent is not None:
        cur = [None if p is None else node.origin[p - 1] for p in cur]
        node = node.parent
    return cur


def newick(node: Node, lengths: bool = True) -> str:
    lab = node.name + (f":{node.d_est:g}" if lengths and node.parent is not None else "")
    if node.is_leaf:
        return lab
    return "(" + ",".join(newick(c, lengths) for c in node.children) + ")" + lab


def clades(root: Node) -> set:
    """Leaf sets below the internal nodes other than the root (rooted comparison)."""
    return {frozenset(l.name for l in n.leaves()) for n in root.walk() if not n.is_leaf and n is not root}


def render(node: Node, indent: str = "", last: bool = True, top: bool = True) -> str:
    seq = " ".join(node.seq)
    if top:
        out = f"{node.name} = {seq}\n"
        child_indent = ""
    else:
        ev = node.events or "no events"
        out = f"{indent}{'└─ ' if last else '├─ '}{node.name} = {seq}    [{ev}; d_est = {node.d_est:g}]\n"
        child_indent = indent + ("   " if last else "│  ")
    for k, c in enumerate(node.children):
        out += render(c, child_indent, k == len(node.children) - 1, False)
    return out
