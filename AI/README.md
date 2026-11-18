# AI: learned reconstruction of the ancestor of two tandem repeat sequences

Experiment of 17 November 2026. A model is trained on simulated minisatellite families
(A; D1, D2) to reconstruct the ancestor A from the two descendants, and is tested on
families it has not seen. The accuracy is measured as in the manuscript
"Tandem repeats ancestor reconstruction and distance" (MPE version) and compared with
the alignment-based reconstruction of that manuscript.

## Contents

| path | content |
|---|---|
| `notebooks/01_generate_dataset.ipynb` | 10,000 simulated families, 80/10/10 split, and the 380 families with k = 2 of the MPE manuscript |
| `notebooks/02_baseline_alignment.ipynb` | alignment-based reconstruction (MPE pipeline) on the test families |
| `notebooks/03_train_models.ipynb` | training of the three candidate models (GPU) |
| `notebooks/04_evaluate.ipynb` | test of the selected model, comparison with A and with the MPE results, distance estimates |
| `tr_ai/` | data generation, models, training, inference and evaluation |
| `tr_aligner/` | the asymmetric alignment and the reconstruction pipeline of the MPE manuscript (copy) |
| `data/mpe_k2_results.csv` | recorded per-family results of the MPE manuscript for k = 2 |
| `tests/test_smoke.py` | end-to-end test on a tiny data set (CPU, about a minute) |
| `ai_ancestor_experiment.tex` | log of the experiment |
| `build_notebooks.py` | regenerates the notebooks |

## Order of running (Google Colab)

1. `01_generate_dataset` (CPU, a few minutes).
2. `02_baseline_alignment` (CPU, about 3 hours in one session) and `03_train_models` (GPU) at the same time.
3. `04_evaluate` (GPU for the reconstruction, CPU for the distance estimates).

All notebooks write to `MyDrive/tandem_repeats/ai_experiment` and are resumable.

## Parallel use

* Every CPU stage uses all cores of its session and works shard by shard; a finished shard is never recomputed.
* The same notebook can be opened in several sessions with `PART = 0, 1, ..., N_PARTS - 1`; the sessions share the Drive folder.
* The three models can be trained in three GPU sessions, one model per session (`MODELS = ["aligntag"]`, and so on).

## Models

| name | model | output |
|---|---|---|
| `aligntag` | BiLSTM over the columns of a symmetric alignment of D1 and D2 | per column: keep the symbol of D1, keep the symbol of D2, or no copy in A |
| `bilstm` | BiLSTM encoder, LSTM decoder with attention | A, token by token |
| `transformer` | encoder-decoder Transformer | A, token by token |

The model with the lowest mean relative error on the validation set is taken to the test.
