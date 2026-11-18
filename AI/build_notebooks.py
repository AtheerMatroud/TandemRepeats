"""Build the four Colab notebooks (run: python build_notebooks.py)."""
import pathlib
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code

OUT = pathlib.Path(__file__).resolve().parent / "notebooks"
OUT.mkdir(exist_ok=True)

SETUP = '''#@title Setup: fetch the code and choose where results are kept
REPO_URL = "https://github.com/AtheerMatroud/TandemRepeats"  #@param {type:"string"}
SAVE_TO_DRIVE = True  #@param {type:"boolean"}

import os, sys, subprocess, zipfile, io
if not os.path.exists("/content/TandemRepeats/AI/tr_ai"):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "/content/TandemRepeats"])
if not os.path.exists("/content/TandemRepeats/AI/tr_ai"):
    from google.colab import files
    print("The AI folder was not found in the repository. Upload AI.zip instead.")
    up = files.upload(); name = next(iter(up))
    os.makedirs("/content/TandemRepeats", exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(up[name])) as z: z.extractall("/content/TandemRepeats")
AI = "/content/TandemRepeats/AI"
sys.path.insert(0, AI)
os.environ["PYTHONPATH"] = AI + ":" + os.environ.get("PYTHONPATH", "")
if SAVE_TO_DRIVE:
    from google.colab import drive; drive.mount("/content/drive")
    ROOT = "/content/drive/MyDrive/tandem_repeats/ai_experiment"
else:
    ROOT = "/content/ai_experiment"
os.makedirs(ROOT, exist_ok=True)
import json, glob, time, numpy as np, pandas as pd
print("code:", AI, "| results:", ROOT, "| CPU cores:", os.cpu_count())
'''

PARALLEL = '''#@title Parallel settings
# One session already uses every CPU core. To go faster, open this notebook in several
# Colab sessions at once and give each a different PART (0, 1, ..., N_PARTS - 1).
# All sessions write to the same Drive folder; finished shards are skipped, so an
# interrupted session can simply be run again.
PART = 0     #@param {type:"integer"}
N_PARTS = 1  #@param {type:"integer"}
'''

# ----------------------------------------------------------------------------- 01
nb1 = [md('''# 1. Simulated data set: 10,000 families (A; D1, D2)

Each family is generated exactly as for the MPE manuscript (`tr_aligner.simulate.TCBBSimulator`):

* ancestor A of about 50s copies of a random 11-mer, s = 2 to 20, carrying 15 motif variants;
* two descendants D1 and D2, each 5t events from A, t = 1 to 20, at deletion : substitution : duplication = 1 : 5 : 20, duplications of 1 to 5 copies and deletions of 1 to 3 copies;
* (s, t) drawn uniformly from the 19 × 20 grid for every family.

The families are split 80 / 10 / 10 into training, validation and test sets, stratified by event density (low if 5t < √(2 n(A) ln 2)). The 380 families with k = 2 of the MPE manuscript are regenerated as a second, independent test set (same seed, hence the same sequences as in the manuscript).

Runtime: CPU only, a few minutes.'''),
       code(SETUP), code(PARALLEL),
       code('''#@title Generate the 10,000 families (parallel, resumable)
N_FAMILIES = 10000  #@param {type:"integer"}
BASE_SEED = 2026    #@param {type:"integer"}
from tr_ai import simdata
t0 = time.time()
simdata.generate(ROOT, n=N_FAMILIES, base_seed=BASE_SEED, shard_size=100, part=PART, n_parts=N_PARTS)
simdata.generate_mpe_grid(ROOT)
print(f"done in {(time.time() - t0) / 60:.1f} min")
'''),
       code('''#@title Split 80 / 10 / 10 (stratified by event density) and summarise
fams = simdata.load(f"{ROOT}/families/shard_*.jsonl.gz")
assert len(fams) == N_FAMILIES, f"{len(fams)} of {N_FAMILIES} families present: let every PART finish first"
splits = simdata.make_splits(fams)
json.dump(splits, open(f"{ROOT}/splits.json", "w"))
where = {i: k for k, v in splits.items() for i in v}
df = pd.DataFrame([{"split": where[f["id"]], "low": f["low"], "s": f["s"], "t": f["t"], "n(A)": len(f["A"]),
                    "n(D1)": len(f["D1"]), "n(D2)": len(f["D2"]), "variants": len(f["variants"]),
                    "A-only variants": len(f["variants"]) - f["nD"], "d_hist": np.mean(f["d_hist"])} for f in fams])
display(df.groupby(["split", "low"]).agg(families=("s", "size"), mean_nA=("n(A)", "mean"), mean_nD=("n(D1)", "mean"),
                                         mean_variants=("variants", "mean"), mean_d_hist=("d_hist", "mean")).round(1))
print("families whose ancestor has a variant absent from both descendants:", int((df["A-only variants"] > 0).sum()))
'''),
       code('''#@title Check: the regenerated MPE families are those of the manuscript
from tr_ai import evaluate as ev
mpe = simdata.load(f"{ROOT}/mpe_grid/s*.jsonl.gz")
rec = ev.mpe_frame(f"{AI}/data/mpe_k2_results.csv").set_index("id")
bad = [f["id"] for f in mpe if len(f["A"]) != rec.loc[f["id"], "len_A"] or abs(f["d_hist"][0] - rec.loc[f["id"], "d_hist_0"]) > 1e-9]
print(len(mpe), "MPE families regenerated;", len(bad), "differ from the recorded results")
assert not bad
''')]

# ----------------------------------------------------------------------------- 02
nb2 = [md('''# 2. Baseline: the alignment-based reconstruction of the MPE manuscript on the test families

For every family of the test set the ancestor is reconstructed from D1 and D2 by the pipeline of the MPE manuscript (`reconstruct_multi`: the two sibling alignments, choice of direction, majority vote), run exactly as in `tcbb_exp1_cell`. Recorded per family: Â, e(A, Â), n(Â) − n(A), and for each descendant d_est(Â, Dᵢ) and d_est(A, Dᵢ) under the match-reward and the edit-weight scorings, with d_hist(A, Dᵢ).

Runtime: CPU only. A family takes from under a second (s = 2) to about a minute (s = 20); the 1,000 test families take roughly 3 hours in one session, proportionally less with parallel sessions. No GPU is needed, so this notebook can run while the models train.'''),
       code(SETUP), code(PARALLEL),
       code('''#@title Run the baseline on the test set (parallel, resumable)
from tr_ai import simdata, evaluate as ev
fams = {f["id"]: f for f in simdata.load(f"{ROOT}/families/shard_*.jsonl.gz")}
splits = json.load(open(f"{ROOT}/splits.json"))
test = sorted((fams[i] for i in splits["test"]), key=lambda f: -len(f["A"]))
print(len(test), "test families")
ev.run_sharded(ev.baseline_family, test, f"{ROOT}/baseline_test", shard_size=10, part=PART, n_parts=N_PARTS)
'''),
       code('''#@title Summary (merges every finished shard)
base = pd.DataFrame(ev.read_shards(f"{ROOT}/baseline_test"))
print(len(base), "of", len(test), "test families done")
print("Accuracy of the reconstructed ancestor, alignment-based pipeline, test set"); display(ev.accuracy_table(base))
print("For reference: the k = 2 families of the MPE manuscript (recorded results)")
display(ev.accuracy_table(ev.mpe_frame(f"{AI}/data/mpe_k2_results.csv")))
base.drop(columns=["Ahat"]).to_csv(f"{ROOT}/baseline_test.csv", index=False)
''')]

# ----------------------------------------------------------------------------- 03
nb3 = [md('''# 3. Training of the three candidate models (GPU)

Given D1 and D2, each model reconstructs A:

| name | model | task |
|---|---|---|
| `aligntag` | alignment-guided tagger (BiLSTM over the columns of a symmetric alignment of D1 and D2) | per column: A keeps the symbol of D1, keeps the symbol of D2, or has no copy there |
| `bilstm` | BiLSTM encoder, LSTM decoder with attention | [D1 ; SEP ; D2] → A, token by token |
| `transformer` | encoder-decoder Transformer | [D1 ; SEP ; D2] → A, token by token |

Variants are numbered within each family in order of first appearance, and every token carries the nucleotides of its variant, so that a model can see which variants are one substitution apart. Training families are augmented by exchanging D1 and D2 and, for the sequence-to-sequence models, by reversing all three sequences.

Model selection uses the validation set only: the mean relative error ε(A, Â) = e(A, Â) / n(A).

**Parallel use.** Set `MODELS` to one name per Colab session to train the three models at the same time (three GPU sessions), or leave the list to train them one after another. Training is checkpointed after every epoch and resumes if the session is interrupted. `STAGE = "pilot"` trains on 2,000 families for a few epochs for a first comparison; `STAGE = "full"` uses the whole training set.

Set the runtime to GPU (Runtime → Change runtime type) before running.'''),
       code(SETUP),
       code('''#@title Parameters
MODELS = ["aligntag", "bilstm", "transformer"]  # one name per session to train in parallel
STAGE = "full"  #@param ["pilot", "full"]
import torch
from tr_ai import simdata
from tr_ai.train import train
from tr_ai.models import DEFAULTS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "(no GPU: training will be very slow)")
fams = {f["id"]: f for f in simdata.load(f"{ROOT}/families/shard_*.jsonl.gz")}
splits = json.load(open(f"{ROOT}/splits.json"))
train_fams = [fams[i] for i in splits["train"]]; val_fams = [fams[i] for i in splits["val"]]
OVERRIDES = {}
if STAGE == "pilot":
    train_fams = train_fams[:2000]
    OVERRIDES = {m: {"epochs": 8} for m in MODELS}
print(len(train_fams), "training and", len(val_fams), "validation families")
'''),
       code('''#@title Train (resumable)
for name in MODELS:
    out = f"{ROOT}/models/{STAGE}/{name}"
    if os.path.exists(f"{out}/done.json"):
        print(name, "already trained:", json.load(open(f"{out}/done.json"))["best"]); continue
    best = train(name, train_fams, val_fams, out, device, overrides=OVERRIDES.get(name), workers=os.cpu_count())
    print(name, "best on validation:", best)
'''),
       code('''#@title Comparison of the candidates on the validation set
import matplotlib.pyplot as plt
rows = []
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for p in sorted(glob.glob(f"{ROOT}/models/{STAGE}/*/log.csv")):
    name = p.split("/")[-2]; log = pd.read_csv(p)
    ax[0].plot(log.epoch, log.val_loss, label=name)
    v = log.dropna(subset=["val_eps"]); ax[1].plot(v.epoch, 100 * v.val_eps, marker="o", label=name)
    d = f"{ROOT}/models/{STAGE}/{name}/done.json"
    if os.path.exists(d):
        b = json.load(open(d)); rows.append({"model": name, "parameters (M)": round(b["parameters"] / 1e6, 1), "best epoch": b["best"]["epoch"],
                                              "val families": b["best"]["n_val"], "exact (%)": round(100 * b["best"]["exact"], 1),
                                              "mean e": round(b["best"]["edit"], 1), "mean eps (%)": round(100 * b["best"]["eps"], 2),
                                              "mean n(Ahat)-n(A)": round(b["best"]["excess"], 1),
                                              "minutes": round(log.seconds.sum() / 60, 1)})
ax[0].set_title("validation loss"); ax[1].set_title("validation mean relative error (%)"); ax[1].set_yscale("log")
for a in ax: a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend()
plt.tight_layout(); plt.savefig(f"{ROOT}/models/{STAGE}_curves.png", dpi=150); plt.show()
if rows:
    cmp = pd.DataFrame(rows).sort_values("mean eps (%)"); display(cmp); cmp.to_csv(f"{ROOT}/models/{STAGE}_comparison.csv", index=False)
''')]

# ----------------------------------------------------------------------------- 04
nb4 = [md('''# 4. Test of the selected model and comparison with the MPE manuscript

1. The model with the lowest validation error is selected (or the one named in `MODEL`).
2. It reconstructs Â from (D1, D2) for the 1,000 test families and for the 380 families with k = 2 of the MPE manuscript. The true ancestor is not seen.
3. **Comparison with the original ancestor A:** exact recovery, e(A, Â), ε(A, Â) and n(Â) − n(A), by event density, as in the MPE manuscript.
4. **Comparison with the MPE result:** on the test set against the alignment-based pipeline run on the same families (notebook 2), and on the 380 MPE families against the results recorded for the manuscript, family by family.
5. **Distance estimates:** d_est(Â, Dᵢ) from the model's ancestor under the match-reward and the edit-weight scorings, against d_hist(A, Dᵢ); sign counts and summary statistics of δ_rel.

Steps 1 to 4 need the GPU for a few minutes. Step 5 is CPU work (about 1 to 2 hours in one session) and is parallel and resumable in the same way as notebook 2.'''),
       code(SETUP), code(PARALLEL),
       code('''#@title Select the model and reconstruct the ancestors of the test families
MODEL = "auto"  #@param ["auto", "aligntag", "bilstm", "transformer"]
import torch
from tr_ai import simdata, evaluate as ev
from tr_ai.infer import load_model, predict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cands = {p.split("/")[-2]: json.load(open(p))["best"] for p in glob.glob(f"{ROOT}/models/full/*/done.json")}
display(pd.DataFrame(cands).T.sort_values("eps"))
name = min(cands, key=lambda k: cands[k]["eps"]) if MODEL == "auto" else MODEL
print("selected model:", name)
fams = {f["id"]: f for f in simdata.load(f"{ROOT}/families/shard_*.jsonl.gz")}
splits = json.load(open(f"{ROOT}/splits.json"))
SETS = {"test": [fams[i] for i in splits["test"]], "mpe": simdata.load(f"{ROOT}/mpe_grid/s*.jsonl.gz")}
PRED = {}
for key, fs in SETS.items():
    path = f"{ROOT}/pred_{name}_{key}.json"
    if not os.path.exists(path):
        model, _ = load_model(name, f"{ROOT}/models/full/{name}/ckpt_best.pt", device)
        t0 = time.time(); preds = predict(model, fs, device, progress=True)
        json.dump({"ids": [f["id"] for f in fs], "Ahat": preds, "seconds": time.time() - t0}, open(path, "w"))
    PRED[key] = json.load(open(path))["Ahat"]
    print(key, len(PRED[key]), "ancestors reconstructed")
ACC = {key: pd.DataFrame([ev.accuracy_row(f, p) for f, p in zip(SETS[key], PRED[key])]) for key in SETS}
'''),
       code('''#@title Accuracy against the true ancestor, and comparison with the alignment-based pipeline
base_test = pd.DataFrame(ev.read_shards(f"{ROOT}/baseline_test"))
base_mpe = ev.mpe_frame(f"{AI}/data/mpe_k2_results.csv")
BASE = {"test": base_test, "mpe": base_mpe}
TITLE = {"test": "test set (1,000 families)", "mpe": "the 380 families with k = 2 of the MPE manuscript"}
tables = {}
for key in SETS:
    common = set(ACC[key].id) & set(BASE[key].id)
    if len(common) < len(ACC[key]): print(f"note: baseline available for {len(common)} of {len(ACC[key])} families so far; tables use those")
    A, B = ACC[key][ACC[key].id.isin(common)], BASE[key][BASE[key].id.isin(common)]
    a = ev.accuracy_table(A).assign(method=f"learned model ({name})")
    b = ev.accuracy_table(B).assign(method="alignment-based (MPE)")
    tables[key] = pd.concat([a, b]).set_index(["method", "event density"])
    print("Accuracy of the reconstructed ancestor:", TITLE[key]); display(tables[key])
    pr = pd.DataFrame({d: ev.paired(A[m(A)], B) for d, m in (("low", lambda x: x.low == 1), ("high", lambda x: x.low == 0),
                                                              ("all", lambda x: x.low >= 0))}).T
    print("Paired comparison of eps on the same families (a = learned model, b = alignment-based)"); display(pr.round(4))
    tables[key].to_csv(f"{ROOT}/table_accuracy_{key}.csv")
'''),
       code('''#@title Figures: error by t and by s, excess length, heatmaps
import matplotlib.pyplot as plt
for key in SETS:
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for df, lab in ((ACC[key], f"learned model ({name})"), (BASE[key], "alignment-based (MPE)")):
        g = df.groupby("t"); ax[0].plot(g.eps.mean().index, 100 * g.eps.mean().values, marker="o", label=lab)
        g = df.groupby("s"); ax[1].plot(g.eps.mean().index, 100 * g.eps.mean().values, marker="o", label=lab)
        g = df.groupby("t"); ax[2].plot(g.excess.mean().index, g.excess.mean().values, marker="o", label=lab)
    ax[0].set_title("mean relative error (%) against t"); ax[0].set_xlabel("t (5t events per descendant)")
    ax[1].set_title("mean relative error (%) against s"); ax[1].set_xlabel("s (about 50s copies in A)")
    ax[2].set_title("mean n(Ahat) - n(A) against t"); ax[2].set_xlabel("t")
    for a in ax: a.grid(alpha=.3); a.legend()
    fig.suptitle(TITLE[key]); plt.tight_layout(); plt.savefig(f"{ROOT}/fig_accuracy_{key}.png", dpi=150); plt.show()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    vmax = 100 * max(ACC[key].groupby(["s", "t"]).eps.mean().max(), BASE[key].groupby(["s", "t"]).eps.mean().max())
    for a, (df, lab) in zip(ax, ((ACC[key], f"learned model ({name})"), (BASE[key], "alignment-based (MPE)"))):
        h = 100 * df.pivot_table(index="t", columns="s", values="eps", aggfunc="mean")
        im = a.imshow(h.values, origin="lower", aspect="auto", vmin=0, vmax=vmax, extent=[h.columns.min() - .5, h.columns.max() + .5, h.index.min() - .5, h.index.max() + .5])
        ss = np.linspace(2, 20, 100); a.plot(ss, np.sqrt(2 * 50 * ss * np.log(2)) / 5, color="orange")
        a.set_ylim(0.5, 20.5); a.set_title(lab); a.set_xlabel("s")
    ax[0].set_ylabel("t"); fig.colorbar(im, ax=ax, label="mean relative error (%)")
    fig.suptitle(TITLE[key] + ": low event density lies below the orange curve")
    plt.savefig(f"{ROOT}/fig_heatmap_{key}.png", dpi=150, bbox_inches="tight"); plt.show()
'''),
       code('''#@title Distance estimates from the model's ancestor (CPU, parallel, resumable)
for key in SETS:
    items = sorted(zip(SETS[key], PRED[key]), key=lambda x: -len(x[0]["A"]))
    ev.run_sharded(ev.model_distance_family, items, f"{ROOT}/dist_{name}_{key}", shard_size=10, part=PART, n_parts=N_PARTS)
'''),
       code('''#@title delta_rel: sign counts and summary statistics (format of the MPE manuscript)
out = []
for key in SETS:
    mod = pd.DataFrame(ev.read_shards(f"{ROOT}/dist_{name}_{key}"))
    print(f"\\n=== {TITLE[key]}: {len(mod)} of {len(SETS[key])} families done ===")
    B = BASE[key][BASE[key].id.isin(set(mod.id))]; mod = mod[mod.id.isin(set(B.id))]
    src = {f"learned model ({name})": ev.edges_frame(mod, "hat"), "alignment-based (MPE)": ev.edges_frame(B, "hat"),
           "true ancestor A": ev.edges_frame(B, "true")}
    for sc in ev.SCORINGS:
        for who, e in src.items():
            sign, stats = ev.drel_tables(e, sc)
            both = sign.merge(stats, on="event density").assign(set=key, scoring=ev.LABEL[sc], ancestor=who)
            out.append(both)
        print(f"delta_rel under the {ev.LABEL[sc]} scoring")
        display(pd.concat(out[-len(src):]).set_index(["ancestor", "event density"]).drop(columns=["set", "scoring"]))
    mod.drop(columns=["Ahat"]).to_csv(f"{ROOT}/model_{name}_{key}.csv", index=False)
pd.concat(out).to_csv(f"{ROOT}/table_delta_rel.csv", index=False)
ACC["test"].to_csv(f"{ROOT}/accuracy_{name}_test.csv", index=False); ACC["mpe"].to_csv(f"{ROOT}/accuracy_{name}_mpe.csv", index=False)
print("All result files are in", ROOT)
''')]

for fname, cells, gpu in (("01_generate_dataset.ipynb", nb1, False), ("02_baseline_alignment.ipynb", nb2, False),
                          ("03_train_models.ipynb", nb3, True), ("04_evaluate.ipynb", nb4, True)):
    nb = new_notebook(); nb.cells = cells
    nb.metadata["colab"] = {"provenance": [], "name": fname}
    if gpu:
        nb.metadata["colab"]["gpuType"] = "T4"; nb.metadata["accelerator"] = "GPU"
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nbf.write(nb, OUT / fname); print("wrote", OUT / fname)
