"""Step 2: the distances d(i).

d(i) is the distance from site i back to the start of the previous occurrence of the
k-tuple that starts at site i. d(i) = 0 when there is no previous occurrence within
max_d, or when the k-tuple holds an invalid symbol. The sequence is scanned once from
left to right over all overlapping k-tuples: O(n) time, O(4^k) additional space.
Numba is used when it is installed; otherwise the same loop runs in plain Python.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

# provisional parameters of the motif length intervals: (k, largest distance recorded)
TIERS = {"7-100": (4, 2000), "101-500": (6, 10000), "501+": (8, 65535)}


def _scan(code, k, max_d, d, last):
    mask = (1 << (2 * k)) - 1
    h = 0
    valid = 0                      # number of consecutive valid symbols ending at j
    for j in range(code.shape[0]):
        c = code[j]
        if c > 3:
            valid = 0
            h = 0
            continue
        h = ((h << 2) | c) & mask
        valid += 1
        if valid >= k:
            i = j - k + 1
            p = last[h]
            if p >= 0 and i - p <= max_d:
                d[i] = i - p
            last[h] = i


try:
    from numba import njit
    _scan = njit(cache=True)(_scan)
except Exception:                  # numba not installed: plain Python loop
    pass


def distances(code: np.ndarray, k: int, max_d: int) -> np.ndarray:
    """d(i) for every site of an encoded sequence (see ntrscan.fasta.encode)."""
    if not 1 <= k <= 12:
        raise ValueError("k must be between 1 and 12")
    code = np.ascontiguousarray(code, dtype=np.int64)
    d = np.zeros(code.shape[0], dtype=np.int32)
    last = np.full(4 ** k, -1, dtype=np.int64)
    _scan(code, k, max_d, d, last)
    return d


def runs(d: np.ndarray, min_len: int = 1) -> List[Tuple[int, int, int, int]]:
    """Runs of constant non-zero d: (first site, last site, run length, d)."""
    out, n, s = [], len(d), 0
    for i in range(1, n + 1):
        if i == n or d[i] != d[s]:
            if d[s] != 0 and i - s >= min_len:
                out.append((s, i - 1, i - s, int(d[s])))
            s = i
    return out
