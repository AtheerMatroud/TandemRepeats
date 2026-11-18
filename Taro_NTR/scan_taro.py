"""Distance scan (Step 2 of the ntrscan work log) on the taro rDNA regions,
followed by run compression (Step 3) for manual inspection.

d(i) = distance from site i back to the start of the previous occurrence of the
k-tuple that starts at i; 0 if none within max_d or if the tuple holds a
non-ACGT symbol.  Sites are numbered from 0 inside the file.
"""
import sys, collections
import numpy as np

CODE = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def read_fasta(path):
    name, seq = None, []
    for line in open(path):
        line = line.strip()
        if line.startswith('>'):
            name = line[1:]
        elif line:
            seq.append(line.upper())
    return name, ''.join(seq)

def distances(seq, k, max_d):
    n = len(seq)
    d = np.zeros(n, dtype=np.int32)
    last = np.full(4 ** k, -1, dtype=np.int64)
    code = [CODE.get(c, -1) for c in seq]
    mask = 4 ** k - 1
    h, valid = 0, 0            # valid = number of consecutive valid symbols
    for j in range(n):
        c = code[j]
        if c < 0:
            valid, h = 0, 0
            continue
        h = ((h << 2) | c) & mask
        valid += 1
        if valid >= k:
            i = j - k + 1
            p = last[h]
            if p >= 0 and i - p <= max_d:
                d[i] = i - p
            last[h] = i
    return d

def runs(d, min_len):
    out, n, s = [], len(d), 0
    for i in range(1, n + 1):
        if i == n or d[i] != d[s]:
            if d[s] != 0 and i - s >= min_len:
                out.append((s, i - 1, i - s, int(d[s])))
            s = i
    return out

if __name__ == '__main__':
    path, k, max_d, min_run = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    name, seq = read_fasta(path)
    d = distances(seq, k, max_d)
    r = runs(d, min_run)
    out = path.rsplit('.', 1)[0] + f'_k{k}_runs.tsv'
    with open(out, 'w') as f:
        f.write('start\tend\trun_length\td\n')
        for x in r:
            f.write('\t'.join(map(str, x)) + '\n')
    print(name)
    print('length', len(seq), 'N', seq.count('N'), 'k', k, 'max_d', max_d, 'min_run', min_run)
    print('runs kept', len(r), '->', out.split('/')[-1])
    c = collections.Counter(x[3] for x in r)
    sites = collections.Counter()
    for x in r: sites[x[3]] += x[2]
    print('d : runs : sites   (top 25 by sites)')
    for dd, s in sites.most_common(25):
        print(f'{dd:6d} {c[dd]:5d} {s:6d}')
