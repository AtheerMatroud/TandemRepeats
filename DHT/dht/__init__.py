"""Duplication history tree for homologous tandem repeat sequences.

Built on tr_aligner (asymmetric alignment and ancestor reconstruction).
"""
from .tree import (Node, pair_join, distance_matrix, closest_pair_tree, neighbour_joining_tree,
                   annotate_edges, trace_to_root, newick, clades, render)

__all__ = ["Node", "pair_join", "distance_matrix", "closest_pair_tree", "neighbour_joining_tree",
           "annotate_edges", "trace_to_root", "newick", "clades", "render"]
