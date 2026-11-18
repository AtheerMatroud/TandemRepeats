"""Training loop for the three candidate models: resumable, mixed precision on GPU."""
from __future__ import annotations

import copy, csv, json, math, os, random, time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import fastalign as fa
from .data import PAD, Seq2SeqDataset, TagDataset, TokenBatchSampler, collate_seq2seq, collate_tag
from .infer import predict, to_device
from .models import DEFAULTS, build


def accuracy(families: List[dict], preds: List[List[int]]) -> Dict[str, float]:
    e = np.array([fa.edit_distance(p, f["A"]) for f, p in zip(families, preds)])
    n = np.array([len(f["A"]) for f in families])
    return {"exact": float((e == 0).mean()), "edit": float(e.mean()), "eps": float((e / n).mean()),
            "excess": float(np.mean([len(p) - k for p, k in zip(preds, n)]))}


def _loss(model, b):
    logits = model(b)
    if model.kind == "tagger":
        return F.cross_entropy(logits.reshape(-1, 3).float(), b["lab"].reshape(-1), ignore_index=-100)
    tgt = b["tgt"][:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1), ignore_index=PAD)


def train(name: str, train_fams: List[dict], val_fams: List[dict], outdir: str, device,
          overrides: Optional[dict] = None, val_decode_n: int = 100, decode_every: int = 5,
          workers: int = 2, seed: int = 0, log=print) -> dict:
    """Train model `name`; checkpoints and the log go to outdir. Returns the best validation record.

    Selection is by the mean relative error eps(A, A_hat) on the validation families
    (all of them for the tagger; a fixed subset of val_decode_n, decoded every
    decode_every epochs, for the sequence-to-sequence models).
    """
    cfg = copy.deepcopy(DEFAULTS[name])
    for k, v in (overrides or {}).items():
        if k == "model":
            cfg["model"].update(v)
        else:
            cfg[k] = v
    os.makedirs(outdir, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = build(name, **cfg["model"]).to(device)
    tagger = model.kind == "tagger"
    DS, coll = (TagDataset, collate_tag) if tagger else (Seq2SeqDataset, collate_seq2seq)
    tr, va = DS(train_fams, augment=True), DS(val_fams, augment=False)
    dl_tr = DataLoader(tr, batch_sampler=TokenBatchSampler(tr.lengths, cfg["max_tokens"], True), collate_fn=coll,
                       num_workers=workers, pin_memory=device.type == "cuda")
    dl_va = DataLoader(va, batch_sampler=TokenBatchSampler(va.lengths, cfg["max_tokens"], False), collate_fn=coll,
                       num_workers=workers)
    rng = random.Random(1)
    val_sub = val_fams if tagger else rng.sample(val_fams, min(val_decode_n, len(val_fams)))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    total = cfg["epochs"] * len(dl_tr)
    warm = min(cfg["warmup"], max(1, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)
    start, best = 0, {"eps": float("inf")}
    last = f"{outdir}/ckpt_last.pt"
    if os.path.exists(last):
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); start, best = ck["epoch"] + 1, ck["best"]
        log(f"[{name}] resumed after epoch {ck['epoch'] + 1}")
    n_par = sum(p.numel() for p in model.parameters())
    log(f"[{name}] {n_par / 1e6:.1f} M parameters, {len(tr)} training families, {len(dl_tr)} batches per epoch")
    fields = ["epoch", "train_loss", "val_loss", "val_eps", "val_exact", "val_edit", "val_excess", "seconds"]
    logpath = f"{outdir}/log.csv"
    if not os.path.exists(logpath):
        with open(logpath, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
    for epoch in range(start, cfg["epochs"]):
        t0 = time.time(); model.train(); tl, nb = 0.0, 0
        for b in dl_tr:
            b = to_device(b, device)
            with torch.autocast(device.type, enabled=amp):
                loss = _loss(model, b)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item(); nb += 1
        model.eval(); vl, nv = 0.0, 0
        with torch.no_grad():
            for b in dl_va:
                b = to_device(b, device)
                with torch.autocast(device.type, enabled=amp):
                    vl += _loss(model, b).item(); nv += 1
        row = {"epoch": epoch + 1, "train_loss": tl / max(nb, 1), "val_loss": vl / max(nv, 1),
               "val_eps": "", "val_exact": "", "val_edit": "", "val_excess": ""}
        if tagger or (epoch + 1) % decode_every == 0 or epoch + 1 == cfg["epochs"]:
            acc = accuracy(val_sub, predict(model, val_sub, device, cfg["max_tokens"], workers=0))
            row.update(val_eps=acc["eps"], val_exact=acc["exact"], val_edit=acc["edit"], val_excess=acc["excess"])
            if acc["eps"] < best["eps"]:
                best = dict(acc, epoch=epoch + 1, val_loss=row["val_loss"], n_val=len(val_sub))
                torch.save({"model": model.state_dict(), "cfg": cfg, "name": name, "best": best}, f"{outdir}/ckpt_best.pt")
        row["seconds"] = round(time.time() - t0, 1)
        with open(logpath, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "cfg": cfg}, last + ".part")
        os.replace(last + ".part", last)
        eps = f"{100 * row['val_eps']:.2f}%" if row["val_eps"] != "" else "-"
        log(f"[{name}] epoch {epoch + 1:3d}/{cfg['epochs']}  train {row['train_loss']:.4f}  val {row['val_loss']:.4f}  "
            f"val eps {eps}  [{row['seconds']:.0f}s]")
    with open(f"{outdir}/done.json", "w") as f:
        json.dump({"name": name, "best": best, "cfg": cfg, "n_train": len(tr), "parameters": n_par}, f, indent=1)
    return best
