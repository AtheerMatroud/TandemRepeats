"""ntrscan: search of a whole genome for nested tandem repeats (Steps 1 and 2).

Rebuilt on 18 November 2026 from the work log (ntr_genome_search_log.tex) and from
Taro_NTR/scan_taro.py, because the original package files of 11 November 2026 were not kept.

fasta   Step 1: read a FASTA file record by record and encode it
track   Step 2: distance d(i) to the previous occurrence of the k-tuple starting at site i
"""
from .fasta import encode, read_fasta
from .track import TIERS, distances, runs

__all__ = ["encode", "read_fasta", "TIERS", "distances", "runs"]
