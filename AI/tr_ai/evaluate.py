"""Evaluation as in the MPE manuscript.

Accuracy of a reconstructed ancestor A_hat against the true ancestor A:
    e(A, A_hat)    symbol edit distance between the TR maps
    exact          e = 0
    eps            e / n(A)
    excess         n(A_hat) - n(A)
Distance estimates: d_est(A_hat, D_i) and d_est(A, D_i) from the asymmetric alignment
under the match-reward scoring (CSV suffix `paper`) and the edit-weight scoring
(CSV suffix `match0`), against the weight d_hist(A, D_i) of the true history:
    delta_rel = (d_est - d_hist) / d_hist.
The baseline is the alignment-based reconstruction of the MPE manuscript
(tr_aligner.pipeline.reconstruct_multi), run exactly as in tcbb_exp1_cell.
"""
from __future__ import annotations

import csv, glob, json, math, os, time
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import fastalign as fa

SCORINGS = ("paper", "match0")
LABEL = {"paper": "match-reward", "match0": "edit-weight"}


def accuracy_row(fam: dict, Ahat: Sequence[int]) -> dict:
    e = fa.edit_distance(Ahat, fam["A"])
    n = len(fam["A"])
    return {"id": fam["id"], "s": fam["s"], "t": fam["t"], "low": int(fam["low"]), "len_A": n,
            "len_Ahat": len(Ahat), "edit": e, "exact": int(e == 0), "eps": e / n, "excess": len(Ahat) - n}


def _symbols(fam: dict):
    """TR maps and variant table exactly as tcbb_exp1_cell builds them."""
    from tr_aligner.simulate import to_symbols
    v = fam["variants"]
    seqs = [[v[i] for i in fam[k]] for k in ("A", "D1", "D2")]
    syms, variants = to_symbols(seqs)
    sym_of = {nt: s for s, nt in variants.items()}
    id_of = {nt: i for i, nt in enumerate(v)}
    return syms, variants, sym_of, id_of


def _scorings(variants):
    from tr_aligner.pipeline import paper_scoring
    from tr_aligner.scoring import Scoring
    return {"paper": paper_scoring(variants), "match0": Scoring(variants=dict(variants), match_reward=0.0)}


def baseline_family(fam: dict) -> dict:
    """Alignment-based reconstruction of the MPE manuscript and its distance estimates."""
    from tr_aligner.align import align
    from tr_aligner.pipeline import paper_scoring, reconstruct_multi, sibling_scoring
    syms, variants, sym_of, id_of = _symbols(fam)
    Am, Dm = syms[0], syms[1:]
    t0 = time.time()
    Ah = reconstruct_multi(Dm, variants, sib=sibling_scoring(variants), paper=paper_scoring(variants)).ancestor
    secs = time.time() - t0
    row = accuracy_row(fam, [id_of[variants[x]] for x in Ah])
    row["Ahat"] = [id_of[variants[x]] for x in Ah]
    row["seconds"] = secs
    scs = _scorings(variants)
    for i, D in enumerate(Dm):
        row[f"d_hist_{i}"] = fam["d_hist"][i]
        for name, sc in scs.items():
            row[f"d_est_hat_{name}_{i}"] = align(Ah, D, sc).distance
            row[f"d_est_true_{name}_{i}"] = align(Am, D, sc).distance
    return row


def model_distance_family(args) -> dict:
    """d_est(A_hat, D_i) for an ancestor reconstructed by a model."""
    fam, Ahat = args
    from tr_aligner.align import align
    syms, variants, sym_of, id_of = _symbols(fam)
    Ah = [sym_of[fam["variants"][i]] for i in Ahat]
    row = accuracy_row(fam, Ahat)
    row["Ahat"] = list(map(int, Ahat))
    scs = _scorings(variants)
    for i, D in enumerate(syms[1:]):
        row[f"d_hist_{i}"] = fam["d_hist"][i]
        for name, sc in scs.items():
            row[f"d_est_hat_{name}_{i}"] = align(Ah, D, sc).distance if Ah else float("nan")
    return row


def _run_shard(args):
    func, path, items = args
    if os.path.exists(path):
        return path
    rows = [func(x) for x in items]
    with open(path + ".part", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(path + ".part", path)
    return path


def run_sharded(func, items: list, outdir: str, shard_size: int = 10, n_proc: Optional[int] = None,
                part: int = 0, n_parts: int = 1, verbose: bool = True) -> None:
    """Apply func to every item, shard by shard, in parallel and resumably.

    Items are ordered by cost (longest first) and dealt to shards so that every shard
    is a mix; a notebook copy with part = i of n_parts takes every n_parts-th shard.
    """
    import multiprocessing as mp
    os.makedirs(outdir, exist_ok=True)
    n_sh = math.ceil(len(items) / shard_size)
    shards = [items[k::n_sh] for k in range(n_sh)]
    jobs = [(func, f"{outdir}/shard_{k:04d}.jsonl", sh) for k, sh in enumerate(shards) if k % n_parts == part]
    t0 = time.time()
    with mp.Pool(n_proc or os.cpu_count()) as pool:
        for i, _ in enumerate(pool.imap_unordered(_run_shard, jobs)):
            if verbose:
                print(f"  {i + 1}/{len(jobs)} shards  [{(time.time() - t0) / 60:.1f} min]", flush=True)


def read_shards(outdir: str) -> List[dict]:
    rows = []
    for p in sorted(glob.glob(f"{outdir}/shard_*.jsonl")):
        with open(p) as f:
            rows.extend(json.loads(line) for line in f)
    return rows


# ---------------------------------------------------------------------------
# tables in the format of the MPE manuscript
# ---------------------------------------------------------------------------

def accuracy_table(df, by_density: bool = True):
    """exact (count, %), mean and median e, mean eps (%), mean excess: low / high / all."""
    import pandas as pd
    out = []
    groups = [("low", df[df.low == 1]), ("high", df[df.low == 0]), ("all", df)] if by_density else [("all", df)]
    for name, g in groups:
        if len(g) == 0:
            continue
        out.append({"event density": name, "families": len(g), "exact": int(g.exact.sum()),
                    "exact (%)": round(100 * g.exact.mean(), 1), "mean e": round(g.edit.mean(), 1),
                    "median e": float(g.edit.median()), "mean eps (%)": round(100 * g.eps.mean(), 1),
                    "mean n(Ahat)-n(A)": round(g.excess.mean(), 1)})
    return pd.DataFrame(out)


def edges_frame(df, who: str = "hat"):
    """One row per ancestor-descendant pair with delta_rel under both scorings."""
    import pandas as pd
    parts = []
    for i in (0, 1):
        d = pd.DataFrame({"id": df["id"], "s": df.s, "t": df.t, "low": df.low, "d_hist": df[f"d_hist_{i}"]})
        for sc in SCORINGS:
            col = f"d_est_{who}_{sc}_{i}"
            if col in df:
                d[f"drel_{sc}"] = (df[col] - d.d_hist) / d.d_hist
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def drel_tables(edges, scoring: str = "paper"):
    """Sign counts and summary statistics of delta_rel (Tables of the MPE manuscript)."""
    import pandas as pd
    col = f"drel_{scoring}"
    sign, stats = [], []
    for name, g in (("low", edges[edges.low == 1]), ("high", edges[edges.low == 0]), ("all", edges)):
        v = g[col].dropna()
        if len(v) == 0:
            continue
        z = (v.abs() < 1e-9)
        sign.append({"event density": name, "count": len(v), "<0": int(((v < 0) & ~z).sum()), "=0": int(z.sum()),
                     ">0": int(((v > 0) & ~z).sum())})
        q = np.percentile(v, [0, 25, 50, 75, 100])
        stats.append({"event density": name, "mean": v.mean(), "sd": v.std(), "p0": q[0], "p25": q[1], "p50": q[2],
                      "p75": q[3], "p100": q[4]})
    return pd.DataFrame(sign), pd.DataFrame(stats).round(2)


def paired(df_a, df_b, col: str = "eps"):
    """Paired comparison of two methods on the same families (a minus b)."""
    from scipy.stats import wilcoxon
    m = df_a[["id", col]].merge(df_b[["id", col]], on="id", suffixes=("_a", "_b"))
    d = m[f"{col}_a"] - m[f"{col}_b"]
    try:
        p = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
    except Exception:
        p = float("nan")
    return {"families": len(m), "mean difference": float(d.mean()), "a better (%)": float(100 * (d < 0).mean()),
            "equal (%)": float(100 * (d == 0).mean()), "b better (%)": float(100 * (d > 0).mean()), "wilcoxon p": p}


def mpe_frame(csv_path: str):
    """The recorded k = 2 results of the MPE manuscript, one row per family, in the
    column layout of baseline_family (per-cell files of Experiment 1, m = 2)."""
    import pandas as pd
    from .simdata import is_low_density
    d = pd.read_csv(csv_path)
    rows = []
    for (s, t), g in d.groupby(["s", "t"]):
        g = g.sort_values("desc")
        r0 = g.iloc[0]
        row = {"id": f"mpe_s{s}_t{t}", "s": int(s), "t": int(t), "low": int(is_low_density(int(t), int(r0.len_A))),
               "len_A": int(r0.len_A), "len_Ahat": int(r0.len_Ahat), "edit": int(r0.edit), "exact": int(r0.exact),
               "eps": r0.edit / r0.len_A, "excess": int(r0.len_Ahat - r0.len_A)}
        for i, (_, r) in enumerate(g.iterrows()):
            row[f"d_hist_{i}"] = r.d_hist
            for sc in SCORINGS:
                row[f"d_est_hat_{sc}_{i}"] = r[f"d_est_hat_{sc}"]
                row[f"d_est_true_{sc}_{i}"] = r[f"d_est_true_{sc}"]
        rows.append(row)
    return pd.DataFrame(rows)
