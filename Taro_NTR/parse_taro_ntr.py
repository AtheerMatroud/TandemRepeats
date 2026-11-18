"""Manual check of the taro NTR: parse the locus into copies of x (11 bp) and
X (48 bp) by dynamic programming on the edit distance.  A base that fits
neither motif is skipped at cost 1."""
import sys
from scan_taro import read_fasta

def ed(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def parse(reg, x, X):
    n = len(reg); INF = 10 ** 9
    f = [INF] * (n + 1); back = [None] * (n + 1); f[0] = 0
    for j in range(1, n + 1):
        f[j] = f[j - 1] + 1; back[j] = (j - 1, '.', 1)
        for mot, lab, lo, hi in ((x, 'x', len(x) - 1, len(x) + 1), (X, 'X', len(X) - 2, len(X) + 2)):
            for L in range(lo, hi + 1):
                if j - L >= 0:
                    c = ed(reg[j - L:j], mot)
                    if c <= 0.3 * len(mot) and f[j - L] + c < f[j]:
                        f[j] = f[j - L] + c; back[j] = (j - L, lab, c)
    out = []; j = n
    while j > 0:
        i, lab, c = back[j]; out.append((i, j, lab, c)); j = i
    return out[::-1]

if __name__ == '__main__':
    path, a, b, off = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    name, seq = read_fasta(path)
    x0 = 'TCGCACAGCCG'
    U = seq[off:off + 59]; UU = U + U
    p = min(range(59), key=lambda q: sum(c1 != c2 for c1, c2 in zip(UU[q:q + 11], x0)))
    X = UU[p + 11:p + 59]
    print('x =', x0); print('X =', X, len(X)); print('d(x, X window best) =', min(ed(X[i:i+11], x0) for i in range(38)))
    segs = parse(seq[a:b], x0, X)
    # compress
    comp = []; 
    for i, j, lab, c in segs:
        if comp and comp[-1][0] == lab: comp[-1][1] += 1; comp[-1][2] += c; comp[-1][4] = a + j - 1
        else: comp.append([lab, 1, c, a + i, a + j - 1])
    nx = sum(1 for s in segs if s[2] == 'x'); nX = sum(1 for s in segs if s[2] == 'X')
    core = [c for c in comp if not (c[0] == '.' )]
    print('copies of x:', nx, ' copies of X:', nX)
    print('map (label^count[start-end, edits]):')
    print(' '.join(f"{c[0]}^{c[1]}[{c[3]}-{c[4]},{c[2]}]" if c[0] != '.' else f"gap{c[1]}[{c[3]}-{c[4]}]" for c in comp))
