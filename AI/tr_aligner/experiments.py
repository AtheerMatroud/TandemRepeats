"""Experiments for the Colab notebooks (27 September 2026).

Experiment 1  one ancestor, m independent descendants (m = 2..6), k events per edge.
Experiment 2  two-level tree: A1 -> (P1, P2), P1 -> (L1, L2), P2 -> (L3, L4), k events
              per edge. All four leaves are used with equal weight at every node
              (sibling structure is taken as unknown): one parent estimate P_hat from
              the four leaves, scored against P1 and P2; then A1 from the four leaves
              plus P_hat. References: each parent from its own two leaves (pairwise),
              A1 from its two true children, A1 from the four leaves alone.

Reporting per cell, following Section 5 of the TCBB paper:
  exact        share of families whose reconstructed node equals the true node
  edit         mean symbol edit distance between reconstructed and true node
  delta_rel    (d_est - d_hist) / d_hist per descendant edge, where d_est is the
               paper's distance estimate from the reconstructed ancestor to the
               descendant and d_hist the weight of the true event history
               (1 + 10(l-1) per l-copy duplication, 20 l |X| per l-copy deletion,
               50 per substituted nucleotide); reported as the fraction with
               delta_rel = 0, the fraction with delta_rel <= 0, mean and s.d.,
               both for the reconstructed ancestor and for the true ancestor.
Per-family rows are written to CSV as each cell completes.
"""

from __future__ import annotations

import csv, itertools, os, statistics, time
from typing import Dict, List, Optional, Sequence, Tuple

from .align import align
from .ancestor import symbol_edit_distance as ed
from .pipeline import paper_scoring, reconstruct_multi, reconstruct_pairwise
from .simulate import Simulator, THESIS_VARIANTS, history_weight


def estimate_scorings(variants: Dict[str, str]):
    """Two scorings for d_est: the paper's (match +50|a|) and the corrected one (match 0),
    under which the alignment minimises Equation (3.1) directly (log, Finding 1)."""
    from .scoring import Scoring
    return {"paper": paper_scoring(variants), "match0": Scoring(variants=dict(variants), match_reward=0.0)}


def _summ(vals: List[float]) -> Dict[str, float]:
    v = [x for x in vals if x == x]
    if not v:
        return {"n": 0}
    return {"n": len(v), "frac_zero": sum(abs(x) < 1e-9 for x in v) / len(v),
            "frac_le0": sum(x <= 1e-9 for x in v) / len(v),
            "mean": statistics.fmean(v), "sd": statistics.pstdev(v) if len(v) > 1 else 0.0}


class Recorder:
    def __init__(self, path: Optional[str], fields: List[str]):
        self.path, self.fields = path, fields
        self.rows: List[dict] = []
        if path and not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def add(self, row: dict) -> None:
        self.rows.append(row)
        if self.path:
            with open(self.path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writerow(row)


# ---------------------------------------------------------------------------
# Experiment 1
# ---------------------------------------------------------------------------

def experiment1_cell(m: int, k: int, n: int = 1000, length: int = 20, seed: int = 1,
                     variants: Dict[str, str] = THESIS_VARIANTS,
                     csv_path: Optional[str] = None, verbose: bool = True) -> dict:
    sim = Simulator(variants, seed=seed * 1000 + m * 100 + k)
    scs = estimate_scorings(variants)
    rec = Recorder(csv_path, ["m", "k", "family", "true_A", "Ahat", "exact", "edit", "desc", "d_hist",
                              "d_est_hat_paper", "d_est_true_paper", "d_est_hat_match0", "d_est_true_match0"])
    exact = 0; edits = []
    dr = {f"{who}/{sc}": [] for who in ("Ahat", "trueA") for sc in scs}
    t0 = time.time()
    for fam in range(n):
        A, Ds, hists = sim.family(length, m, k)
        Ah = reconstruct_multi(Ds, variants).ancestor
        e = ed(A, Ah); exact += (e == 0); edits.append(e)
        for i, (D, h) in enumerate(zip(Ds, hists)):
            dh = history_weight(h, variants)
            row = {"m": m, "k": k, "family": fam, "true_A": A, "Ahat": "".join(Ah), "exact": int(e == 0),
                   "edit": e, "desc": i, "d_hist": dh}
            for sc_name, sc in scs.items():
                for who, anc in (("Ahat", Ah), ("trueA", A)):
                    de = align(anc, D, sc).distance
                    row[f"d_est_{'hat' if who == 'Ahat' else 'true'}_{sc_name}"] = de
                    dr[f"{who}/{sc_name}"].append((de - dh) / dh if dh else float("nan"))
            rec.add(row)
    out = {"m": m, "k": k, "n": n, "exact": exact / n, "edit": statistics.fmean(edits),
           "delta_rel": {x: _summ(v) for x, v in dr.items()}, "seconds": time.time() - t0}
    if verbose:
        print(f"m={m} k={k}: exact {out['exact']:.1%}  edit {out['edit']:.2f}   [{out['seconds']:.0f}s]")
        for x, sm in out["delta_rel"].items():
            print(f"      delta_rel {x:13s} =0 {sm['frac_zero']:5.1%}  <=0 {sm['frac_le0']:5.1%}  mean {sm['mean']:+.2f}  sd {sm['sd']:.2f}")
    return out


# ---------------------------------------------------------------------------
# Experiment 2
# ---------------------------------------------------------------------------

def tree2_family(sim: Simulator, length: int, k: int):
    A = sim.random_ancestor(length)
    P, hP = zip(*[sim.descend(A, k) for _ in range(2)])
    L, hL = [], []
    for p in P:
        for _ in range(2):
            d, h = sim.descend(p, k); L.append(d); hL.append(h)
    return A, list(P), list(hP), L, hL


def experiment2_cell(k: int, n: int = 1000, length: int = 20, seed: int = 2,
                     variants: Dict[str, str] = THESIS_VARIANTS,
                     csv_path: Optional[str] = None, verbose: bool = True) -> dict:
    sim = Simulator(variants, seed=seed * 1000 + k)
    nodes = ["Phat_vs_P1", "Phat_vs_P2", "P1_own_leaves", "P2_own_leaves",
             "A1_leaves_plus_Phat", "A1_leaves_only", "A1_true_children"]
    rec = Recorder(csv_path, ["k", "family", "node", "true", "estimate", "exact", "edit"])
    edge = Recorder(csv_path.replace(".csv", "_edges.csv") if csv_path else None,
                    ["k", "family", "edge", "ancestor_used", "d_hist", "d_est", "delta_rel"])
    ex = {x: 0 for x in nodes}; edits = {x: [] for x in nodes}
    scs = estimate_scorings(variants)
    dr = {f"{x}/{sc}": [] for x in ("A1hat->P", "trueA1->P", "Phat->L", "trueP->L") for sc in scs}
    t0 = time.time()
    for fam in range(n):
        A, P, hP, L, hL = tree2_family(sim, length, k)
        Phat = reconstruct_multi(L, variants).ancestor
        own = [reconstruct_pairwise(L[0], L[1], variants).ancestor, reconstruct_pairwise(L[2], L[3], variants).ancestor]
        A_lp = reconstruct_multi(L + [list(Phat)], variants).ancestor
        A_l = reconstruct_multi(L, variants).ancestor
        A_tc = reconstruct_pairwise(P[0], P[1], variants).ancestor
        est = {"Phat_vs_P1": (P[0], Phat), "Phat_vs_P2": (P[1], Phat), "P1_own_leaves": (P[0], own[0]),
               "P2_own_leaves": (P[1], own[1]), "A1_leaves_plus_Phat": (A, A_lp), "A1_leaves_only": (A, A_l),
               "A1_true_children": (A, A_tc)}
        for name, (t, e_) in est.items():
            e = ed(t, e_); ex[name] += (e == 0); edits[name].append(e)
            rec.add({"k": k, "family": fam, "node": name, "true": t, "estimate": "".join(e_), "exact": int(e == 0), "edit": e})
        # paper-style edge reporting: A1 -> P_i with A1 estimated from leaves + Phat, and with true A1
        for i in range(2):
            dh = history_weight(hP[i], variants)
            for label, anc in (("A1hat->P", A_lp), ("trueA1->P", A)):
                for sc_name, sc in scs.items():
                    de = align(anc, P[i], sc).distance
                    r = (de - dh) / dh if dh else float("nan"); dr[f"{label}/{sc_name}"].append(r)
                    edge.add({"k": k, "family": fam, "edge": f"A1->P{i + 1}", "ancestor_used": f"{label}/{sc_name}", "d_hist": dh, "d_est": de, "delta_rel": r})
        for j in range(4):
            dh = history_weight(hL[j], variants)
            for label, anc in (("Phat->L", Phat), ("trueP->L", P[j // 2])):
                for sc_name, sc in scs.items():
                    de = align(anc, L[j], sc).distance
                    r = (de - dh) / dh if dh else float("nan"); dr[f"{label}/{sc_name}"].append(r)
                    edge.add({"k": k, "family": fam, "edge": f"P{j // 2 + 1}->L{j + 1}", "ancestor_used": f"{label}/{sc_name}", "d_hist": dh, "d_est": de, "delta_rel": r})
    out = {"k": k, "n": n, "seconds": time.time() - t0,
           "nodes": {x: {"exact": ex[x] / n, "edit": statistics.fmean(edits[x])} for x in nodes},
           "delta_rel": {x: _summ(v) for x, v in dr.items()}}
    if verbose:
        print(f"k={k}  [{out['seconds']:.0f}s]")
        for x in nodes:
            print(f"   {x:22s} exact {out['nodes'][x]['exact']:6.1%}  edit {out['nodes'][x]['edit']:.2f}")
        for x, sm in out["delta_rel"].items():
            print(f"   delta_rel {x:18s} =0 {sm['frac_zero']:5.1%}  <=0 {sm['frac_le0']:5.1%}  mean {sm['mean']:+.2f}  sd {sm['sd']:.2f}")
    return out


# ---------------------------------------------------------------------------
# TCBB Section 5 protocol (27 September 2026): full 380-cell grid, one family per cell
# ---------------------------------------------------------------------------

from .simulate import TCBBSimulator, to_symbols
from .pipeline import sibling_scoring

GRID_S = list(range(2, 21))     # ancestor ~ 50 s copies: 100 .. 1000
GRID_T = list(range(1, 21))     # 5 t events per edge: 5 .. 100


def _scorings_for(variants):
    from .scoring import Scoring
    return {"paper": paper_scoring(variants), "match0": Scoring(variants=dict(variants), match_reward=0.0)}


def tcbb_exp1_cell(s: int, t: int, m: int, seed: int = 11, csv_path: Optional[str] = None,
                   verbose: bool = True) -> dict:
    """One family of the TCBB grid: ancestor ~50s copies, m descendants each 5t events away."""
    sim = TCBBSimulator(seed=seed * 100_000 + m * 10_000 + s * 100 + t)
    A, Ds, hists, weights = sim.family(s, t, m)
    syms, variants = to_symbols([A] + Ds)
    Am, Dm = syms[0], syms[1:]
    t0 = time.time()
    r = reconstruct_multi(Dm, variants, sib=sibling_scoring(variants), paper=paper_scoring(variants))
    Ah = r.ancestor
    e = ed(Am, Ah)
    scs = _scorings_for(variants)
    rec = Recorder(csv_path, ["s", "t", "m", "len_A", "len_Ahat", "n_variants", "exact", "edit", "desc", "len_D",
                              "d_hist", "d_est_hat_paper", "d_est_true_paper", "d_est_hat_match0", "d_est_true_match0", "seconds"])
    out = {"s": s, "t": t, "m": m, "len_A": len(Am), "len_Ahat": len(Ah), "n_variants": len(variants),
           "exact": int(e == 0), "edit": e, "edges": []}
    for i, (D, w) in enumerate(zip(Dm, weights)):
        row = {"s": s, "t": t, "m": m, "len_A": len(Am), "len_Ahat": len(Ah), "n_variants": len(variants),
               "exact": int(e == 0), "edit": e, "desc": i, "len_D": len(D), "d_hist": w, "seconds": 0.0}
        for name, sc in scs.items():
            row[f"d_est_hat_{name}"] = align(Ah, D, sc).distance
            row[f"d_est_true_{name}"] = align(Am, D, sc).distance
        out["edges"].append(row)
    out["seconds"] = time.time() - t0
    for row in out["edges"]:
        row["seconds"] = out["seconds"]; rec.add(row)
    if verbose:
        dr = [(row["d_est_hat_paper"] - row["d_hist"]) / row["d_hist"] for row in out["edges"]]
        print(f"s={s:2d} t={t:2d} m={m}: |A|={len(Am):4d} |Ahat|={len(Ah):4d} exact={e == 0} edit={e:3d} "
              f"mean delta_rel(Ahat,paper)={statistics.fmean(dr):+.2f}  [{out['seconds']:.0f}s]")
    return out


def tcbb_exp2_cell(s: int, t: int, seed: int = 12, csv_path: Optional[str] = None, verbose: bool = True,
                   edges_csv_path: Optional[str] = None) -> dict:
    """One family of the two-level tree on the TCBB grid: A1 -> (P1, P2) -> four leaves, 5t events per edge."""
    sim = TCBBSimulator(seed=seed * 100_000 + s * 100 + t)
    A = sim.ancestor(s)
    P, hP, wP = zip(*[sim.descend(A, t) for _ in range(2)])
    L, hL, wL = [], [], []
    for p in P:
        for _ in range(2):
            d, h, w = sim.descend(p, t); L.append(d); hL.append(h); wL.append(w)
    syms, variants = to_symbols([A] + list(P) + L)
    Am, Pm, Lm = syms[0], syms[1:3], syms[3:]
    sib, pap = sibling_scoring(variants), paper_scoring(variants)
    t0 = time.time()
    Phat = reconstruct_multi(Lm, variants, sib=sib, paper=pap).ancestor
    own = [reconstruct_pairwise(Lm[0], Lm[1], variants, sib=sib, paper=pap).ancestor,
           reconstruct_pairwise(Lm[2], Lm[3], variants, sib=sib, paper=pap).ancestor]
    A_lp = reconstruct_multi(Lm + [list(Phat)], variants, sib=sib, paper=pap).ancestor
    A_l = reconstruct_multi(Lm, variants, sib=sib, paper=pap).ancestor
    A_tc = reconstruct_pairwise(Pm[0], Pm[1], variants, sib=sib, paper=pap).ancestor
    est = {"Phat_vs_P1": (Pm[0], Phat), "Phat_vs_P2": (Pm[1], Phat), "P1_own_leaves": (Pm[0], own[0]),
           "P2_own_leaves": (Pm[1], own[1]), "A1_leaves_plus_Phat": (Am, A_lp), "A1_leaves_only": (Am, A_l),
           "A1_true_children": (Am, A_tc)}
    secs = time.time() - t0
    rec = Recorder(csv_path, ["s", "t", "node", "len_true", "len_est", "exact", "edit", "seconds"])
    out = {"s": s, "t": t, "len_A": len(Am), "n_variants": len(variants), "nodes": {}, "edges": [], "seconds": secs}
    for name, (tr, es) in est.items():
        e = ed(tr, es); out["nodes"][name] = {"exact": int(e == 0), "edit": e, "len_true": len(tr), "len_est": len(es)}
        rec.add({"s": s, "t": t, "node": name, "len_true": len(tr), "len_est": len(es), "exact": int(e == 0), "edit": e, "seconds": secs})
    scs = _scorings_for(variants)
    edge = Recorder(edges_csv_path or (csv_path.replace(".csv", "_edges.csv") if csv_path else None),
                    ["s", "t", "edge", "ancestor_used", "scoring", "d_hist", "d_est"])
    for i in range(2):
        for label, anc in (("A1hat->P", A_lp), ("trueA1->P", Am)):
            for name, sc in scs.items():
                de = align(anc, Pm[i], sc).distance
                row = {"s": s, "t": t, "edge": f"A1->P{i + 1}", "ancestor_used": label, "scoring": name, "d_hist": wP[i], "d_est": de}
                out["edges"].append(row); edge.add(row)
    for j in range(4):
        for label, anc in (("Phat->L", Phat), ("trueP->L", Pm[j // 2])):
            for name, sc in scs.items():
                de = align(anc, Lm[j], sc).distance
                row = {"s": s, "t": t, "edge": f"P{j // 2 + 1}->L{j + 1}", "ancestor_used": label, "scoring": name, "d_hist": wL[j], "d_est": de}
                out["edges"].append(row); edge.add(row)
    if verbose:
        print(f"s={s:2d} t={t:2d}: |A1|={len(Am):4d}  " + "  ".join(f"{k} e={v['edit']}" for k, v in out["nodes"].items()) + f"  [{secs:.0f}s]")
    return out


def summarise_edges(rows: List[dict], key_fields=("m",), est_col="d_est_hat_paper") -> Dict[tuple, dict]:
    """delta_rel statistics per key over per-edge rows with d_hist and the given d_est column."""
    groups: Dict[tuple, List[float]] = {}
    for r in rows:
        if r["d_hist"] > 0:
            groups.setdefault(tuple(r[k] for k in key_fields), []).append((r[est_col] - r["d_hist"]) / r["d_hist"])
    return {k: _summ(v) for k, v in groups.items()}
