"""Tests of Steps 1 and 2 (run: python tests/test_step1_step2.py, or pytest)."""
import gzip, os, random, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from ntrscan import distances, encode, read_fasta, runs


def slow(seq, k, max_d):
    seq = seq.upper(); last = {}; d = [0] * len(seq)
    for i in range(len(seq) - k + 1):
        t = seq[i:i + k]
        if set(t) <= set("ACGT"):
            if t in last and i - last[t] <= max_d:
                d[i] = i - last[t]
            last[t] = i
    return d


def test_encoding():
    assert encode("ACGTacgtN-").tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 4, 4]


def test_fasta_plain_and_gz():
    text = ">r1 first\nACGT\nacgt\n\n>r2\nNNAC\n"
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "x.fa"); open(p, "w").write(text)
        g = os.path.join(tmp, "x.fa.gz"); gzip.open(g, "wt").write(text)
        for path in (p, g):
            recs = list(read_fasta(path))
            assert [n for n, _ in recs] == ["r1 first", "r2"]
            assert recs[0][1].tolist() == [0, 1, 2, 3, 0, 1, 2, 3] and recs[1][1].tolist() == [4, 4, 0, 1]


def test_overlapping_tuples_by_hand():
    # ACACAC, k=2: AC at 0,2,4 and CA at 1,3
    assert distances(encode("ACACAC"), 2, 100).tolist() == [0, 0, 2, 2, 2, 0]
    # AAAA, k=2: tuples overlap in k-1 sites, none skipped
    assert distances(encode("AAAA"), 2, 100).tolist() == [0, 1, 1, 0]


def test_invalid_lowercase_max_distance():
    assert distances(encode("ACGTNACGT"), 3, 100).tolist() == [0, 0, 0, 0, 0, 5, 5, 0, 0]
    assert distances(encode("acgtACGT"), 4, 100).tolist() == [0, 0, 0, 0, 4, 0, 0, 0]
    assert distances(encode("ACGT" + "T" * 10 + "ACGT"), 4, 5)[14] == 0


def test_against_slow_implementation():
    rng = random.Random(1)
    for _ in range(200):
        seq = "".join(rng.choice("ACGTACGTACGTN") for _ in range(rng.randint(1, 300)))
        for k in (1, 2, 3, 4, 6, 8):
            md = rng.choice([5, 50, 1000])
            assert distances(encode(seq), k, md).tolist() == slow(seq, k, md)


def test_exact_ntr_distances():
    # x^3 X x^2 X x^4 X x^2 : inside the NTR every distance is m, m+l or l+s_i*m
    x, X = "ACGGTCA", "GACCTTTGGC"
    m, l = len(x), len(X)
    rng = random.Random(7)
    flank = lambda: "".join(rng.choice("ACGT") for _ in range(12))
    seq = flank() + x * 3 + X + x * 2 + X + x * 4 + X + x * 2 + flank()
    r = runs(distances(encode(seq), 4, 2000), 4)
    seen = {dd for s, e, n, dd in r if s >= 12 and e < len(seq) - 12}
    assert m in seen and m + l in seen and l + 2 * m in seen and l + 4 * m in seen
    assert all(dd == m or dd % m == l % m for dd in seen)
    # the copies of X give runs of length l + k - 1 = 13
    assert sorted(dd for s, e, n, dd in r if n == 13) == [l + 2 * m, l + 4 * m]


def test_embedded_ntrs_in_random_sequence():
    rng = random.Random(3)
    rand = lambda n: "".join(rng.choice("ACGT") for _ in range(n))
    fracs = []
    for _ in range(100):
        m, l = rng.randint(7, 40), rng.randint(7, 40)
        x, X = rand(m), rand(l)
        ntr = x * 4 + X + x * 3 + X + x * 5
        seq = rand(500) + ntr + rand(500)
        d = distances(encode(seq), 4, 2000)[500:500 + len(ntr)]
        fracs.append(np.mean(d[m: 4 * m - 3] == m))    # inside the first run x^4
    # below 1 only because a 4-tuple may occur twice inside x (log, Table of signal fractions)
    assert np.mean(fracs) > 0.8 and min(fracs) > 0.4


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)
