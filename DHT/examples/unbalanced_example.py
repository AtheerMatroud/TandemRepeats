"""Second hand-traceable example: unbalanced tree with four descendants and an outgroup.

True history (one event per edge):
    O = a b c
      OUT = e b c                  S1   (a -> e), the outgroup
      R = a b b c                  R2   (duplication of b)
        D1 = a b b f               S4   (c -> f)
        X = a b b c c              R4   (duplication of c)
          D2 = d b b c c           S1   (a -> d)
          Y = a b b b c c          R2   (duplication of b)
            D3 = a b b b c c c     R5   (duplication of c)
            D4 = a b b b b c c     R2   (duplication of b)
Input to the method: D1, D2, D3, D4 and the outgroup OUT.
Expected: tree (D1, (D2, (D3, D4))) with root R = a b b c.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from dht import neighbour_joining_tree, closest_pair_tree, annotate_edges, trace_to_root, newick, clades, render

VARIANTS = {"a": "CATGGT", "b": "TATGGA", "c": "CACGCT", "d": "TATGGT", "e": "CAAGGT", "f": "CACGGT"}
TRUE = {"R": "abbc", "X": "abbcc", "Y": "abbbcc"}
LEAVES = {"D1": "abbf", "D2": "dbbcc", "D3": "abbbccc", "D4": "abbbbcc"}
OUTGROUP = {"OUT": "ebc"}
TRUE_CLADES = {frozenset({"D3", "D4"}), frozenset({"D2", "D3", "D4"})}


def show(title, root, log):
    print("=" * 72); print(title); print("=" * 72)
    for e in log:
        if "dist" in e:
            print("distances:", {f"{a},{b}": d for (a, b), d in e["dist"].items()})
        if "Q" in e:
            print("Q:", {f"{a},{b}": q for (a, b), q in e["Q"].items()})
        if "joined" in e:
            extra = f" -> {e['new']} = {' '.join(e['ancestor'])} (ties: {e['ties']})" if "new" in e else ""
            print(f"step {e['step']}: join {e['joined']}{extra}")
    annotate_edges(root, VARIANTS)
    print(); print(render(root))
    print("Newick:", newick(root) + ";")
    print("clades correct:", clades(root) == TRUE_CLADES, "  root correct:", root.seq == list(TRUE["R"]))
    print("total weight of the tree:", sum(n.d_est for n in root.walk() if n.parent is not None))
    for n in root.walk():
        if n.parent is not None:
            print(f"\n{n.parent.name} -> {n.name}: {n.events or 'no events'}  (d_est = {n.d_est:g})")
            print(n.alignment.pretty())
    print()
    for l in root.leaves():
        print(f"{l.name}: copies {' '.join(l.seq)} descend from root copies {trace_to_root(l)}")
    print()


if __name__ == "__main__":
    ingroup = {k: list(v) for k, v in LEAVES.items()}
    with_out = dict(ingroup, **{k: list(v) for k, v in OUTGROUP.items()})
    log = []
    show("Neighbour joining, rooted by the outgroup", neighbour_joining_tree(with_out, VARIANTS, log=log, outgroup="OUT"), log)
    log = []
    show("Neighbour joining, rooted on the last edge joined (no outgroup)", neighbour_joining_tree(ingroup, VARIANTS, log=log), log)
    log = []
    show("Closest pair (no outgroup)", closest_pair_tree(ingroup, VARIANTS, log=log), log)
