"""Checks on the unbalanced example with an outgroup (run: python tests/test_unbalanced_example.py, or pytest)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))

from dht import neighbour_joining_tree, annotate_edges, trace_to_root, clades
from unbalanced_example import VARIANTS, LEAVES, OUTGROUP, TRUE_CLADES

IN = {k: list(v) for k, v in LEAVES.items()}
ALL = dict(IN, **{k: list(v) for k, v in OUTGROUP.items()})


def test_outgroup_rooting_recovers_topology():
    t = annotate_edges(neighbour_joining_tree(ALL, VARIANTS, outgroup="OUT"), VARIANTS)
    assert clades(t) == TRUE_CLADES
    assert sorted(l.name for l in t.leaves()) == ["D1", "D2", "D3", "D4"]      # outgroup removed
    assert len(t.seq) == 4                                                     # root has four motif copies
    assert {l.name: trace_to_root(l) for l in t.leaves()} == {
        "D1": [1, 2, 3, 4], "D2": [1, 2, 3, 4, 4], "D3": [1, 2, 2, 3, 4, 4, 4], "D4": [1, 2, 2, 2, 3, 4, 4]}


def test_recorded_root_variant():
    # recorded behaviour, not the desired one: the true root is a b b c; the variant d of D2
    # is carried up to the root because no correction is applied (ruling of 18 Nov 2026)
    t = neighbour_joining_tree(ALL, VARIANTS, outgroup="OUT")
    assert t.seq == list("dbbc")


def test_last_edge_rooting_fails_without_outgroup():
    assert clades(neighbour_joining_tree(IN, VARIANTS)) != TRUE_CLADES


if __name__ == "__main__":
    test_outgroup_rooting_recovers_topology(); test_recorded_root_variant()
    test_last_edge_rooting_fails_without_outgroup(); print("ok")
