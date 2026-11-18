"""Parse every taro IGS locus into copies of x (11 bp) and X (48 bp) and save
the NTR of each allele (the core of adjoining motif copies) as FASTA."""
import sys
from parse_taro_ntr import parse
x = 'TCGCACAGCCG'
X = 'TTCTGGGCAAAACGGCTGGGCGACGTGCTGGACTAGCCAGCTGGTTCG'
def fasta(path):
    name, out = None, {}
    for line in open(path):
        line = line.strip()
        if line.startswith('>'): name = line[1:]; out[name] = []
        elif line: out[name].append(line.upper())
    return {k: ''.join(v) for k, v in out.items()}
recs = fasta(sys.argv[1])
fo = open('taro_NTR_alleles.fasta', 'w'); ft = open('taro_NTR_alleles_maps.tsv', 'w')
ft.write('allele\tsource\tntr_start_in_record\tntr_end_in_record\tntr_length\tcopies_X\tcopies_x\tedits\tx_counts_between_X\tmap\n')
n = 0
for name, seq in recs.items():
    segs = parse(seq, x, X)
    comp = []
    for i, j, lab, c in segs:
        if comp and comp[-1][0] == lab: comp[-1][1] += 1; comp[-1][2] += c; comp[-1][4] = j
        else: comp.append([lab, 1, c, i, j])
    best, cur = [], []
    for c in comp + [['.', 99, 0, 0, 0]]:
        if c[0] == '.' and c[1] > 15:
            if sum(b[4] - b[3] for b in cur) > sum(b[4] - b[3] for b in best): best = cur
            cur = []
        else: cur.append(c)
    while best and best[0][0] == '.': best = best[1:]
    while best and best[-1][0] == '.': best = best[:-1]
    if not best: continue
    a, b = best[0][3], best[-1][4]
    nX = sum(c[1] for c in best if c[0] == 'X'); nx = sum(c[1] for c in best if c[0] == 'x')
    ed = sum(c[2] for c in best if c[0] != '.') + sum(c[1] for c in best if c[0] == '.')
    counts, run = [], 0
    for c in best:
        if c[0] == 'x': run += c[1]
        elif c[0] == 'X': counts.append(run); run = 0
    counts.append(run)
    mp = ' '.join(f"{c[0]}^{c[1]}" if c[0] != '.' else f"[{c[1]}]" for c in best)
    n += 1; al = f"A{n:02d}"
    fo.write(f">{al}|{name}|ntr={a+1}-{b}|len={b-a}|X={nX}|x={nx}\n")
    s = seq[a:b]
    for i in range(0, len(s), 70): fo.write(s[i:i+70] + '\n')
    ft.write('\t'.join(map(str, [al, name, a + 1, b, b - a, nX, nx, ed, ','.join(map(str, counts)), mp])) + '\n')
    print(al, name.split('|')[0], name.split('|')[1], 'len', b - a, 'X', nX, 'x', nx, 'ed', ed, '|', ','.join(map(str, counts)))
fo.close(); ft.close()
