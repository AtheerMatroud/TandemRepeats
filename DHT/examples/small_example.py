"""Small hand-traceable example: balanced tree with four descendants.

True history (one event per edge):
    R = a b c
      X = a b b c        R2       (duplication of b)
        D1 = a b b b c   R2 (or R3)
        D2 = a b b f     S4       (c -> f, Hamming distance 1)
      Y = a b c c        R3       (duplication of c)
        D3 = a b c c c   R3 (or R4)
        D4 = d b c c     S1       (a -> d, Hamming distance 1)
Input to the method: D1, D2, D3, D4 only.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from dht import (closest_pair_tree, neighbour_joining_tree, annotate_edges, trace_to_root,
                 newick, clades, render)

VARIANTS = {"a": "CATGGT", "b": "TATGGA", "c": "CACGCT", "d": "TATGGT", "e": "CAAGGT", "f": "CACGGT"}
TRUE = {"R": "abc", "X": "abbc", "Y": "abcc"}
LEAVES = {"D1": "abbbc", "D2": "abbf", "D3": "abccc", "D4": "dbcc"}
TRUE_CLADES = {frozenset({"D1", "D2"}), frozenset({"D3", "D4"})}


def report(title, root, log):
    print("=" * 72); print(title); print("=" * 72)
    for e in log:
        if "dist" in e:
            print("distances:", {f"{a},{b}": d for (a, b), d in e["dist"].items()})
        if "ancestors" in e:
            print("pair ancestors:", {f"{a},{b}": " ".join(s) for (a, b), s in e["ancestors"].items()})
        if "Q" in e:
            print("Q:", {f"{a},{b}": q for (a, b), q in e["Q"].items()})
        if "joined" in e:
            extra = f" -> {e['new']} = {' '.join(e['ancestor'])} (ties: {e['ties']})" if "new" in e else ""
            print(f"step {e['step']}: join {e['joined']}{extra}")
    annotate_edges(root, VARIANTS)
    print(); print(render(root))
    print("Newick:", newick(root) + ";")
    print("clades correct:", clades(root) == TRUE_CLADES)
    for n in root.walk():
        if n.parent is not None:
            print(f"\n{n.parent.name} -> {n.name}: {n.events or 'no events'}  (d_est = {n.d_est:g})")
            print(n.alignment.pretty())
            print("origin of each copy in", n.parent.name, ":", n.origin)
    print()
    for l in root.leaves():
        print(f"{l.name}: copies {' '.join(l.seq)} descend from root copies {trace_to_root(l)}")
    print()


if __name__ == "__main__":
    leaves = {k: list(v) for k, v in LEAVES.items()}
    log1, log2 = [], []
    t1 = closest_pair_tree(leaves, VARIANTS, log=log1)
    report("Rule 1: closest pair", t1, log1)
    t2 = neighbour_joining_tree(leaves, VARIANTS, log=log2)
    report("Rule 2: neighbour joining", t2, log2)
    print("same clades under both rules:", clades(t1) == clades(t2))
