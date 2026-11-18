"""End-to-end smoke test on a tiny data set (CPU, about a minute): python tests/test_smoke.py"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tr_ai import simdata, fastalign as fa, evaluate as ev
from tr_ai.train import train, accuracy
from tr_ai.infer import predict, load_model

fams = [f for f in (simdata.make_family(i) for i in range(400)) if f["s"] <= 4][:60]
tr, va = fams[:48], fams[48:]
dev = torch.device("cpu")
small = {"transformer": dict(model=dict(d=64, heads=2, enc_layers=1, dec_layers=1, ff=128), epochs=2, max_tokens=6000),
         "bilstm": dict(model=dict(d=64, layers=1), epochs=2, max_tokens=6000),
         "aligntag": dict(model=dict(d=64, layers=1), epochs=3, max_tokens=6000)}
with tempfile.TemporaryDirectory() as tmp:
    for name, ov in small.items():
        best = train(name, tr, va, f"{tmp}/{name}", dev, overrides=ov, val_decode_n=4, decode_every=2, workers=0)
        model, _ = load_model(name, f"{tmp}/{name}/ckpt_best.pt", dev)
        preds = predict(model, va[:4], dev, workers=0)
        print(name, "best", {k: round(v, 3) for k, v in best.items()}, accuracy(va[:4], preds))
    row = ev.baseline_family(va[0]); print("baseline", {k: v for k, v in row.items() if k != "Ahat"})
    row = ev.model_distance_family((va[0], preds[0])); print("model d_est", {k: v for k, v in row.items() if k != "Ahat"})
print("smoke test passed")
