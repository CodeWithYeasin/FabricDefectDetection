"""Training and inference loops (AMP, warmup+cosine, EMA, early stopping on val macro-F1)."""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import FabricDataset, balanced_sampler, orientation_swap_map
from .losses import build_loss
from .metrics import compute_metrics
from .models import build_model
from .utils import get_logger, save_json

log = get_logger()


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay, self.updates = decay, 0
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))  # warm-up so early epochs aren't stuck at init
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def _cosine(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def _loader(df, cfg, train, classes):
    d = cfg["data"]
    ds = FabricDataset(df, d["img_size"], train=train, geo_mode=cfg["aug"]["geo_mode"],
                       strength=cfg["aug"]["strength"], classes=classes,
                       grayscale=d.get("grayscale", False))
    kw = dict(batch_size=cfg["train"]["batch_size"], num_workers=d.get("num_workers", 4),
              pin_memory=torch.cuda.is_available(), persistent_workers=d.get("num_workers", 4) > 0)
    if train:
        sp = cfg["train"].get("sampler_power", 0.0)
        if sp > 0:
            return DataLoader(ds, sampler=balanced_sampler(ds.ys, sp), drop_last=True, **kw)
        return DataLoader(ds, shuffle=True, drop_last=True, **kw)
    return DataLoader(ds, shuffle=False, **kw)


@torch.no_grad()
def predict(model, loader, device, tta: str = "none", classes=None) -> np.ndarray:
    """tta: none | flip | oa  (oa adds 90/270 rotations with horizontal<->vertical prob swap)."""
    model.eval()
    perm = orientation_swap_map(classes) if classes else None
    out = []
    amp = device.type == "cuda"
    for x, _, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            views = [(x, False)]
            if tta in {"flip", "oa"}:
                views += [(torch.flip(x, [-1]), False), (torch.flip(x, [-2]), False)]
            if tta == "oa":
                views += [(torch.rot90(x, 1, [-2, -1]), True), (torch.rot90(x, 3, [-2, -1]), True)]
            ps = []
            for v, swapped in views:
                p = torch.softmax(model(v).float(), -1)
                if swapped:
                    p = p[:, perm]
                ps.append(p)
        out.append(torch.stack(ps).mean(0).cpu().numpy())
    return np.concatenate(out)


def train_fold(cfg: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame | None,
               out_dir: str | Path, classes: list[str]) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = len(classes)
    m = cfg["model"]
    model = build_model(m["name"], K, pretrained=m.get("pretrained", True),
                        **{k: v for k, v in m.items() if k not in {"name", "pretrained"}}).to(device)
    if cfg["train"].get("channels_last", True) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    t = cfg["train"]
    counts = np.bincount(train_df["y"], minlength=K)
    criterion = build_loss(t["loss"], counts, t.get("label_smoothing", 0.0), t.get("focal_gamma", 2.0)).to(device)
    opt = torch.optim.AdamW(model.param_groups(t["lr"], t.get("backbone_lr_mult", 0.1), t.get("weight_decay", 0.05)))
    base_lrs = [g["lr"] for g in opt.param_groups]
    tl, vl = _loader(train_df, cfg, True, classes), _loader(val_df, cfg, False, classes)
    total = t["epochs"] * len(tl)
    warm = int(t.get("warmup_epochs", 1) * len(tl))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    ema = EMA(model, t.get("ema_decay", 0.999)) if t.get("ema", True) else None

    best, best_ep, bad, step, hist = -1.0, -1, 0, 0, []
    t0 = time.time()
    for ep in range(t["epochs"]):
        model.train()
        run_loss, nb = 0.0, 0
        for x, y, _ in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if device.type == "cuda" and t.get("channels_last", True):
                x = x.contiguous(memory_format=torch.channels_last)
            s = _cosine(step, total, warm)
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * s
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t.get("grad_clip", 1.0))
            scaler.step(opt)
            scaler.update()
            if ema:
                ema.update(model)
            run_loss += loss.item()
            nb += 1
            step += 1
            if t.get("max_steps_per_epoch") and nb >= t["max_steps_per_epoch"]:
                break
        eval_model = ema.ema if ema else model
        pv = predict(eval_model, vl, device)
        mv = compute_metrics(val_df["y"].values, pv, classes)
        hist.append({"epoch": ep, "train_loss": run_loss / max(nb, 1), "val_macro_f1": mv["macro_f1"],
                     "val_acc": mv["accuracy"], "val_nll": mv["nll"], "time_s": time.time() - t0})
        log.info(f"ep {ep:03d} loss {run_loss / max(nb, 1):.4f} val mF1 {mv['macro_f1']:.4f} acc {mv['accuracy']:.4f}")
        score = mv["macro_f1"] - 1e-3 * mv["nll"]  # tie-break on NLL
        if score > best:
            best, best_ep, bad = score, ep, 0
            torch.save({"model": eval_model.state_dict(), "cfg": cfg, "classes": classes, "epoch": ep},
                       out_dir / "best.pt")
        else:
            bad += 1
            if bad >= t.get("patience", 10):
                log.info(f"early stop at epoch {ep}")
                break
    pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)

    ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    res = {"best_epoch": best_ep, "train_time_s": time.time() - t0}
    pv = predict(model, vl, device)  # val probs of the *selected* checkpoint
    tta = cfg.get("eval", {}).get("tta", "none")
    if test_df is not None and len(test_df):
        tel = _loader(test_df, cfg, False, classes)
        pt = predict(model, tel, device, tta=tta, classes=classes)
        res["test"] = compute_metrics(test_df["y"].values, pt, classes)
        pred_df = test_df[["path", "y"]].copy()
        pred_df["pred"] = pt.argmax(1)
        for i, c in enumerate(classes):
            pred_df[f"p_{c}"] = pt[:, i]
        pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
        if tta != "none":
            pn = predict(model, tel, device, tta="none")
            res["test_notta"] = compute_metrics(test_df["y"].values, pn, classes)
            nd = test_df[["path", "y"]].copy()
            nd["pred"] = pn.argmax(1)
            for i, c in enumerate(classes):
                nd[f"p_{c}"] = pn[:, i]
            nd.to_csv(out_dir / "test_predictions_notta.csv", index=False)
    # validation probs are kept for post-hoc temperature scaling / ensemble weighting
    val_df[["path", "y"]].assign(**{f"p_{c}": pv[:, i] for i, c in enumerate(classes)}).to_csv(
        out_dir / "val_predictions.csv", index=False)
    save_json(res, out_dir / "metrics.json")
    return res
