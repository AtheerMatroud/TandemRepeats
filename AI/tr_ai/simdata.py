"""Simulated families (A; D1, D2), generated exactly as in the MPE manuscript.

Every family is produced by tr_aligner.simulate.TCBBSimulator (ancestor of about
50 s copies of an 11-mer with 15 motif variants; two descendants, each 5 t events
away at deletion : substitution : duplication = 1 : 5 : 20). A family is stored as
integer TR maps over its own variant table. Variants are numbered in order of first
appearance in D1, then in D2, then (if any) those that occur in A only; `nD` is the
number of variants that occur in the descendants.
"""
from __future__ import annotations

import glob, gzip, json, math, os, random
from typing import Dict, List, Optional, Sequence

from tr_aligner.simulate import TCBBSimulator

S_RANGE = (2, 20)
T_RANGE = (1, 20)
MPE_SEED = 11          # seed of the k = 2 families of the MPE manuscript


def is_low_density(t: int, n_A: int) -> bool:
    """Low event density: x(D) = 5t < sqrt(2 n(A) ln 2)."""
    return 5 * t < math.sqrt(2 * n_A * math.log(2))


def encode_family(fid: str, s: int, t: int, A: Sequence[str], Ds: Sequence[Sequence[str]],
                  weights: Sequence[float]) -> dict:
    table: Dict[str, int] = {}
    for seq in list(Ds):
        for c in seq:
            if c not in table:
                table[c] = len(table)
    nD = len(table)
    for c in A:
        if c not in table:
            table[c] = len(table)
    variants = [None] * len(table)
    for c, i in table.items():
        variants[i] = c
    return {"id": fid, "s": s, "t": t, "variants": variants, "nD": nD,
            "A": [table[c] for c in A], "D1": [table[c] for c in Ds[0]], "D2": [table[c] for c in Ds[1]],
            "d_hist": [float(w) for w in weights], "low": bool(is_low_density(t, len(A)))}


def make_family(idx: int, base_seed: int = 2026) -> dict:
    """Family number idx of the data set: (s, t) uniform on the MPE grid."""
    rng = random.Random(base_seed * 1_000_003 + idx)
    s, t = rng.randint(*S_RANGE), rng.randint(*T_RANGE)
    sim = TCBBSimulator(seed=base_seed * 10_000_000 + idx)
    A, Ds, _, w = sim.family(s, t, 2)
    return encode_family(f"f{idx:05d}", s, t, A, Ds, w)


def make_mpe_family(s: int, t: int) -> dict:
    """The k = 2 family of cell (s, t) of the MPE manuscript (same seed, same sequences)."""
    sim = TCBBSimulator(seed=MPE_SEED * 100_000 + 2 * 10_000 + s * 100 + t)
    A, Ds, _, w = sim.family(s, t, 2)
    return encode_family(f"mpe_s{s}_t{t}", s, t, A, Ds, w)


def write_shard(path: str, families: List[dict]) -> None:
    tmp = path + ".part"
    with gzip.open(tmp, "wt") as f:
        for fam in families:
            f.write(json.dumps(fam, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def _shard_job(args):
    path, ids, base_seed = args
    if not os.path.exists(path):
        write_shard(path, [make_family(i, base_seed) for i in ids])
    return path


def _mpe_job(args):
    path, cells = args
    if not os.path.exists(path):
        write_shard(path, [make_mpe_family(s, t) for s, t in cells])
    return path


def generate(outdir: str, n: int = 10_000, base_seed: int = 2026, shard_size: int = 100,
             n_proc: Optional[int] = None, part: int = 0, n_parts: int = 1, verbose: bool = True) -> None:
    """Generate the n families into outdir/families, shard by shard, in parallel.

    Parallel within the session (one process per core) and across sessions: a notebook
    copy with part = i of n_parts takes every n_parts-th shard. Finished shards are skipped.
    """
    import multiprocessing as mp
    os.makedirs(f"{outdir}/families", exist_ok=True)
    jobs = []
    for k in range(math.ceil(n / shard_size)):
        if k % n_parts != part:
            continue
        ids = list(range(k * shard_size, min(n, (k + 1) * shard_size)))
        jobs.append((f"{outdir}/families/shard_{k:04d}.jsonl.gz", ids, base_seed))
    with mp.Pool(n_proc or os.cpu_count()) as pool:
        for i, p in enumerate(pool.imap_unordered(_shard_job, jobs)):
            if verbose and (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(jobs)} shards", flush=True)


def generate_mpe_grid(outdir: str, n_proc: Optional[int] = None) -> None:
    import multiprocessing as mp
    os.makedirs(f"{outdir}/mpe_grid", exist_ok=True)
    jobs = [(f"{outdir}/mpe_grid/s{s:02d}.jsonl.gz", [(s, t) for t in range(T_RANGE[0], T_RANGE[1] + 1)])
            for s in range(S_RANGE[0], S_RANGE[1] + 1)]
    with mp.Pool(n_proc or os.cpu_count()) as pool:
        list(pool.imap_unordered(_mpe_job, jobs))


def load(pattern: str) -> List[dict]:
    fams = []
    for p in sorted(glob.glob(pattern)):
        with gzip.open(p, "rt") as f:
            fams.extend(json.loads(line) for line in f)
    return fams


def make_splits(families: List[dict], seed: int = 7, frac=(0.8, 0.1, 0.1)) -> Dict[str, List[str]]:
    """80/10/10 split, stratified by event density."""
    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}
    for low in (True, False):
        ids = sorted(f["id"] for f in families if f["low"] == low)
        rng.shuffle(ids)
        a, b = round(frac[0] * len(ids)), round((frac[0] + frac[1]) * len(ids))
        out["train"] += ids[:a]; out["val"] += ids[a:b]; out["test"] += ids[b:]
    return {k: sorted(v) for k, v in out.items()}
