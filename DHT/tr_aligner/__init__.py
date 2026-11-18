"""Python port of the asymmetric ancestor-descendant alignment for tandem repeats.

Reference: A. Matroud, C. Tuffley and M. Hendy, "An asymmetric alignment
algorithm for estimating ancestor-descendant edit distance for tandem
repeats", IEEE/ACM TCBB, doi 10.1109/TCBB.2021.3059239.

Modules
-------
scoring      scoring matrix sigma (Hamming default) and duplication cost vector delta
levenshtein  Levenshtein distance and scoring for variants of unequal length
align        the dynamic programme (Section 3.2), traceback, events, distance estimate
mapping      variant maps and the wrap-around DP that parses DNA into motif copies
"""

from .scoring import Scoring, delta_vector, hamming
from .levenshtein import levenshtein
from .align import align, Alignment, Event, replay_events, parse_event_string
from .mapping import VariantMap, parse_tandem_repeat

__all__ = [
    "Scoring", "delta_vector", "hamming", "levenshtein",
    "align", "Alignment", "Event", "replay_events", "parse_event_string", "VariantMap", "parse_tandem_repeat",
]
