"""Step 1: reading a FASTA file one record at a time.

Nucleotides are encoded A=0, C=1, G=2, T=3; every other symbol is invalid (4).
Lower case is treated as upper case. Files ending in .gz are read compressed.
"""
from __future__ import annotations

import gzip
from typing import Iterator, Tuple

import numpy as np

INVALID = 4
_TABLE = np.full(256, INVALID, dtype=np.uint8)
for _c, _v in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
    _TABLE[ord(_c)] = _v
    _TABLE[ord(_c.lower())] = _v


def encode(seq) -> np.ndarray:
    """Encode a nucleotide string (str or bytes) as a uint8 array."""
    if isinstance(seq, str):
        seq = seq.encode("ascii", "replace")
    return _TABLE[np.frombuffer(seq, dtype=np.uint8)]


def read_fasta(path: str) -> Iterator[Tuple[str, np.ndarray]]:
    """Yield (name, encoded sequence) for each record; one record is held in memory at a time."""
    opener = gzip.open if str(path).endswith(".gz") else open
    name, parts = None, []
    with opener(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(b">"):
                if name is not None:
                    yield name, encode(b"".join(parts))
                name, parts = line[1:].decode("ascii", "replace"), []
            else:
                parts.append(line)
    if name is not None:
        yield name, encode(b"".join(parts))
