"""Fast CPU unit tests. Run: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fabricnet import CLASSES  # noqa: E402
from fabricnet.data import GeometricAug, make_folds, orientation_swap_map  # noqa: E402
from fabricnet.data.dedup import find_duplicate_groups, resolve_groups  # noqa: E402
from fabricnet.legacy import ScratchGBDT, fit_weights  # noqa: E402
from fabricnet.metrics import compute_metrics  # noqa: E402
from fabricnet.models import SpectralResidual, build_model  # noqa: E402
from fabricnet.stats import bootstrap_ci, friedman_nemenyi, mcnemar  # noqa: E402


def weave(n=128, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    t = 0.5 + 0.3 * np.sin(2 * np.pi * xx / 6) * np.sin(2 * np.pi * yy / 7) + rng.normal(0, 0.02, (n, n))
    return t


def to_tensor(gray):
    rgb = np.repeat(gray[None], 3, 0)[None].astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    return torch.tensor((rgb - mean) / std, dtype=torch.float32)


def test_spectral_residual_localises_a_hole():
    g = weave()
    yy, xx = np.mgrid[0:128, 0:128]
    g[(yy - 90) ** 2 + (xx - 40) ** 2 < 36] = 0.0
    sr = SpectralResidual(scales=(64,), learnable=False, out_size=128).eval()
    with torch.no_grad():
        s = sr(to_tensor(g))[0, 0].numpy()
    peak = np.unravel_index(s.argmax(), s.shape)
    assert abs(peak[0] - 90) < 12 and abs(peak[1] - 40) < 12, peak


def test_orientation_swap():
    perm = orientation_swap_map(CLASSES)
    h, v = CLASSES.index("horizontal"), CLASSES.index("vertical")
    assert perm[h] == v and perm[v] == h
    assert all(perm[i] == i for i in range(len(CLASSES)) if i not in (h, v))


def test_oa_rotation_swaps_label_only_on_odd_turns():
    import random

    img = Image.fromarray((weave() * 255).astype(np.uint8))
    aug = GeometricAug("oa", p_flip=0.0)
    h, v = CLASSES.index("horizontal"), CLASSES.index("vertical")
    seen = set()
    for s in range(40):
        random.seed(s)
        random.random(), random.random()  # the two flip draws inside GeometricAug
        k = random.randint(0, 3)
        random.seed(s)
        _, y = aug(img, h)
        assert y == (v if k % 2 else h)
        seen.add(y)
    assert seen == {h, v}
    naive = GeometricAug("naive_rot", p_flip=0.0)
    assert all(naive(img, h)[1] == h for _ in range(10))


@pytest.mark.parametrize("name,kw", [("spectex_resnet18", {}), ("spectex_resnet18", {"fusion": "concat"}),
                                     ("spectex_resnet18", {"fusion": "none"}), ("resnet18", {})])
def test_models_forward_backward(name, kw):
    m = build_model(name, 6, pretrained=False, **kw)
    x = torch.randn(2, 3, 96, 96)
    out = m(x)
    assert out.shape == (2, 6)
    out.sum().backward()


def test_sga_starts_close_to_identity():
    m = build_model("spectex_resnet18", 6, pretrained=False).eval()
    with torch.no_grad():
        m(torch.randn(1, 3, 96, 96))
    assert m._last_attn.mean().item() < 0.2  # sigmoid(-2) ~ 0.12


def test_dedup_and_groupkfold(tmp_path):
    rows = []
    for c in range(3):
        for i in range(12):
            g = weave(64, seed=100 * c + i)
            p = tmp_path / f"{c}_{i}.png"
            Image.fromarray((g * 255).clip(0, 255).astype(np.uint8)).save(p)
            rows.append({"path": str(p), "label": CLASSES[c], "y": c, "md5": f"{c}_{i}"})
    # an exact copy that must end up in the same group
    Image.open(rows[0]["path"]).save(tmp_path / "dup.png")
    rows.append({"path": str(tmp_path / "dup.png"), "label": CLASSES[0], "y": 0, "md5": "0_0"})
    df, pairs = find_duplicate_groups(pd.DataFrame(rows))
    assert df.loc[0, "group"] == df.loc[len(df) - 1, "group"]
    kept, stats = resolve_groups(df, "keep_all")
    folds = make_folds(kept, n_splits=3, seed=0)
    assert folds.groupby("group")["fold"].nunique().max() == 1


def test_metrics_and_stats():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 6, 300)
    P = np.eye(6)[y] * 0.6 + rng.dirichlet(np.ones(6), 300) * 0.4
    m = compute_metrics(y, P, CLASSES)
    assert 0.9 < m["accuracy"] <= 1 and "macro_auroc" in m and "defect_detection_rate" in m
    v, lo, hi = bootstrap_ci(y, P, lambda a, b: (a == b.argmax(1)).mean(), n_boot=100)
    assert lo <= v <= hi
    assert mcnemar(y, y, y)["p_value"] == 1.0
    fr = friedman_nemenyi(rng.random((5, 4)), list("abcd"))
    assert set(fr["avg_rank"]) == set("abcd")


def test_scratch_gbdt_learns():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 20))
    y = (X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 0.5).astype(int)
    m = ScratchGBDT(n_estimators=15, max_depth=3, seed=0).fit(X, y)
    acc = (m.predict_proba(X).argmax(1) == y).mean()
    assert acc > 0.9
    w, s = fit_weights([m.predict_proba(X)] * 2, y, n_trials=20)
    assert np.isclose(w.sum(), 1)


def test_experiment_registry_configs_resolve():
    import yaml
    from train import experiment_config

    with open(ROOT / "configs/experiments.yaml") as f:
        names = list(yaml.safe_load(f)["experiments"])
    for n in names:
        cfg = experiment_config(n)
        assert cfg["model"]["name"]
    assert experiment_config("abl_no_spectral")["model"]["fusion"] == "none"
    assert experiment_config("abl_aug_naive_rot")["aug"]["geo_mode"] == "naive_rot"


def test_derived_filename_detection():
    from fabricnet.data.index import split_derived

    assert split_derived("11_processed (1)") == ("11", True)
    assert split_derived("14_processed(3)") == ("14", True)
    assert split_derived("exp1_num_249594 - Copy") == ("exp1_num_249594", True)
    assert split_derived("20180531_135032(1)") == ("20180531_135032", True)
    assert split_derived("hole_2018-10-11 13_00_10.730645") == ("hole_2018-10-11 13_00_10.730645", False)
    assert split_derived("0001_000_00") == ("0001_000_00", False)


def test_fold_frames_keeps_derived_out_of_val_and_test():
    from fabricnet.data import fold_frames

    rows = []
    for g in range(60):
        y = g % 3
        rows.append({"path": f"o{g}", "y": y, "group": g, "kind": "original", "fold": g % 5})
        for k in range(3):
            rows.append({"path": f"d{g}_{k}", "y": y, "group": g, "kind": "derived", "fold": g % 5})
    df = pd.DataFrame(rows)
    tr, va, te = fold_frames(df, 0, "fold", 0.2, 0, use_derived=True)
    assert (va["kind"] == "original").all() and (te["kind"] == "original").all()
    assert not set(tr["group"]) & set(te["group"]) and not set(tr["group"]) & set(va["group"])
    assert (tr["kind"] == "derived").sum() == 3 * tr[tr["kind"] == "original"]["group"].nunique()
    tr2, _, _ = fold_frames(df, 0, "fold", 0.2, 0, use_derived=False)
    assert (tr2["kind"] == "original").all()
