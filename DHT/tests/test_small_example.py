"""Checks on the small balanced example (run: python -m pytest tests, or python tests/test_small_example.py)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))

from dht import closest_pair_tree, neighbour_joining_tree, annotate_edges, trace_to_root, clades
from small_example import VARIANTS, LEAVES, TRUE_CLADES

L = {k: list(v) for k, v in LEAVES.items()}


def test_neighbour_joining_recovers_clades_and_root():
    t = annotate_edges(neighbour_joining_tree(L, VARIANTS), VARIANTS)
    assert clades(t) == TRUE_CLADES
    assert t.seq == list("abc")
    trace = {l.name: trace_to_root(l) for l in t.leaves()}
    assert trace == {"D1": [1, 2, 2, 2, 3], "D2": [1, 2, 2, 3], "D3": [1, 2, 3, 3, 3], "D4": [1, 2, 3, 3]}


def test_closest_pair_joins_D1_D3_first():
    # recorded behaviour, not the desired one: two independent duplications (10 + 10 each side)
    # are cheaper than one substitution (50), so D1 and D3 are joined before D1 and D2
    log = []
    closest_pair_tree(L, VARIANTS, log=log)
    assert log[0]["joined"] == ("D1", "D3")


if __name__ == "__main__":
    test_neighbour_joining_recovers_clades_and_root(); test_closest_pair_joins_D1_D3_first(); print("ok")
