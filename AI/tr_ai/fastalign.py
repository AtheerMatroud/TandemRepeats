"""Fast (numba) dynamic programmes used by the learned models and by the evaluation.

nw_align       symmetric global alignment of the TR maps of D1 and D2 (match 0,
               mismatch 20 per differing nucleotide, gap 10 per copy; the symmetric
               analogue of the sibling scoring of the MPE manuscript)
column_labels  for every alignment column, whether the ancestor A keeps the symbol
               of D1, keeps the symbol of D2, or has nothing there (weighted longest
               common subsequence of A and the columns)
edit_distance  symbol edit distance e between two TR maps
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except Exception:                      # pragma: no cover - slow fallback
    def njit(*a, **k):
        def wrap(f):
            return f
        return wrap(a[0]) if a and callable(a[0]) else wrap

DROP, KEEP_X, KEEP_Y = 0, 1, 2


def hamming_matrix(variants) -> np.ndarray:
    arr = np.array([[ord(c) for c in v] for v in variants], dtype=np.int16)
    return (arr[:, None, :] != arr[None, :, :]).sum(-1).astype(np.int32)


@njit(cache=True)
def _nw(x, y, H, gap, mm):
    n, m = len(x), len(y)
    S = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        S[i, 0] = i * gap
    for j in range(1, m + 1):
        S[0, j] = j * gap
    for i in range(1, n + 1):
        xi = x[i - 1]
        for j in range(1, m + 1):
            d = S[i - 1, j - 1] + mm * H[xi, y[j - 1]]
            u = S[i - 1, j] + gap
            l = S[i, j - 1] + gap
            if u < d:
                d = u
            if l < d:
                d = l
            S[i, j] = d
    cx = np.empty(n + m, dtype=np.int32)
    cy = np.empty(n + m, dtype=np.int32)
    k = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and S[i, j] == S[i - 1, j - 1] + mm * H[x[i - 1], y[j - 1]]:
            cx[k] = x[i - 1]; cy[k] = y[j - 1]; i -= 1; j -= 1
        elif i > 0 and S[i, j] == S[i - 1, j] + gap:
            cx[k] = x[i - 1]; cy[k] = -1; i -= 1
        else:
            cx[k] = -1; cy[k] = y[j - 1]; j -= 1
        k += 1
    return cx[:k][::-1].copy(), cy[:k][::-1].copy()


def nw_align(x, y, H, gap: int = 10, mm: int = 20):
    """Columns (cx, cy) of the alignment; -1 marks a gap."""
    return _nw(np.asarray(x, dtype=np.int32), np.asarray(y, dtype=np.int32), H, gap, mm)


@njit(cache=True)
def _labels(cx, cy, A):
    n, m = len(cx), len(A)
    BIG = 10000
    L = np.zeros((n + 1, m + 1), dtype=np.int64)
    for i in range(1, n + 1):
        x, y = cx[i - 1], cy[i - 1]
        bonus = 0
        if x >= 0 and y >= 0:
            bonus = 2 if x == y else 1
        for j in range(1, m + 1):
            b = L[i - 1, j]
            if L[i, j - 1] > b:
                b = L[i, j - 1]
            a = A[j - 1]
            if a == x or a == y:
                c = L[i - 1, j - 1] + BIG + bonus
                if c > b:
                    b = c
            L[i, j] = b
    lab = np.zeros(n, dtype=np.int8)
    i, j = n, m
    while i > 0 and j > 0:
        x, y = cx[i - 1], cy[i - 1]
        a = A[j - 1]
        bonus = 0
        if x >= 0 and y >= 0:
            bonus = 2 if x == y else 1
        if (a == x or a == y) and L[i, j] == L[i - 1, j - 1] + BIG + bonus:
            lab[i - 1] = 1 if a == x else 2
            i -= 1; j -= 1
        elif L[i, j] == L[i - 1, j]:
            i -= 1
        else:
            j -= 1
    return lab, L[n, m] // BIG


def column_labels(cx, cy, A):
    """Labels per column and the number of symbols of A that the columns can express."""
    return _labels(cx, cy, np.asarray(A, dtype=np.int32))


@njit(cache=True)
def _edit(a, b):
    n, m = len(a), len(b)
    prev = np.arange(m + 1).astype(np.int32)
    cur = np.empty(m + 1, dtype=np.int32)
    for i in range(1, n + 1):
        cur[0] = i
        for j in range(1, m + 1):
            c = prev[j - 1] + (0 if a[i - 1] == b[j - 1] else 1)
            if prev[j] + 1 < c:
                c = prev[j] + 1
            if cur[j - 1] + 1 < c:
                c = cur[j - 1] + 1
            cur[j] = c
        prev, cur = cur, prev
    return prev[m]


def edit_distance(a, b) -> int:
    """Symbol edit distance e(a, b) between two TR maps given as integer sequences."""
    return int(_edit(np.asarray(a, dtype=np.int32), np.asarray(b, dtype=np.int32)))


def apply_labels(cx, cy, lab):
    """The ancestor spelled by a labelling of the columns."""
    out = []
    for x, y, l in zip(cx, cy, lab):
        if l == KEEP_X and x >= 0:
            out.append(int(x))
        elif l == KEEP_Y and y >= 0:
            out.append(int(y))
        elif l != DROP:                       # label points at a gap: fall back to the symbol present
            out.append(int(x if x >= 0 else y))
    return out
