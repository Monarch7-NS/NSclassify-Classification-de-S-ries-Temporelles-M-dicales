"""Tests de non-fuite, de forme et de cohérence. Les tests sur données réelles sont ignorés si le cache est absent.

Lancer depuis PROJET/ :  python -m pytest tests -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_processing as dp  # noqa: E402
from src.datasets import ECGData, Preprocessor, crop, prepare  # noqa: E402
from src.losses import FocalLoss, build_loss, positive_weight  # noqa: E402
from src.models import PRESETS, build_model  # noqa: E402

CFG = dp.load_config()
HAVE_CACHE = all(dp.cache_path(CFG, s).exists() for s in dp.SPLIT_NAMES) and dp.SPLITS_PATH.exists()
needs_cache = pytest.mark.skipif(not HAVE_CACHE, reason="cache absent : python -m scripts.build_dataset")


# ---------- étiquettes ----------

def test_label_mapping_strict_v1():
    f = lambda s: dp.label_of(s, CFG)  # noqa: E731
    assert f("N") == 0
    for s in "LRBAaJSVrFejnE":
        assert f(s) == 1, s
    for s in ("/", "f", "Q", "?", "+", "~", "|"):
        assert f(s) is None, s


def test_paced_records_not_in_any_list():
    rec = CFG["records"]
    used = set(rec["ds1"]) | set(rec["ds2"])
    assert used.isdisjoint(rec["paced_excluded"])
    assert len(used) == 44


# ---------- split ----------

@needs_cache
def test_splits_disjoint_and_same_patient_together():
    splits = json.loads(dp.SPLITS_PATH.read_text(encoding="utf-8"))
    dp.assert_split_integrity(splits, CFG)
    where = {s: ("201" in splits[s], "202" in splits[s]) for s in dp.SPLIT_NAMES}
    assert where["test"] == (True, True)
    assert where["train"] == (False, False) and where["val"] == (False, False)


@needs_cache
def test_assert_split_integrity_catches_leak():
    splits = json.loads(dp.SPLITS_PATH.read_text(encoding="utf-8"))
    bad = {**splits, "val": splits["val"] + [splits["train"][0]]}
    with pytest.raises(AssertionError):
        dp.assert_split_integrity(bad, CFG)
    split_pair = {**splits, "train": splits["train"] + ["201"], "test": [r for r in splits["test"] if r != "201"]}
    with pytest.raises(AssertionError):
        dp.assert_split_integrity(split_pair, CFG)


@needs_cache
def test_cached_windows_belong_to_their_split():
    splits = json.loads(dp.SPLITS_PATH.read_text(encoding="utf-8"))
    for s in dp.SPLIT_NAMES:
        with np.load(dp.cache_path(CFG, s)) as d:
            assert set(d["record_id"].tolist()) <= set(splits[s])
            assert (d["split"] == s).all()
            assert d["X_raw"].shape[1] == 360 and d["X_bp"].shape == d["X_raw"].shape
            assert set(np.unique(d["y"])) <= {0, 1}
            assert not np.isin(d["symbol"], CFG["label_mapping"]["excluded_beats"]).any()


# ---------- prétraitement ----------

def test_preprocessor_refuses_non_train_fit():
    X = np.random.randn(10, 216).astype(np.float32)
    for split in ("val", "test"):
        with pytest.raises(ValueError):
            Preprocessor("P2").fit(X, X, split)


def test_global_scalers_use_train_statistics_only():
    rng = np.random.default_rng(0)
    train = rng.normal(0.0, 1.0, (200, 216)).astype(np.float32)
    test = rng.normal(5.0, 3.0, (200, 216)).astype(np.float32)
    p = Preprocessor("P2").fit(train, train, "train")
    assert abs(p.stats["mean"] - train.mean()) < 1e-6
    out = p.transform(test, test)
    assert abs(out.mean() - (test.mean() - train.mean()) / p.stats["std"]) < 1e-3   # pas re-centré sur le test
    q = Preprocessor("P3").fit(train, train, "train")
    assert q.stats["median"] == pytest.approx(float(np.median(train)), abs=1e-5)


def test_per_beat_zscore_is_independent_of_other_beats():
    X = np.random.randn(20, 216).astype(np.float32) * 3 + 2
    p = Preprocessor("P1").fit(X, X, "train")
    a = p.transform(X, X)
    b = p.transform(X[:5], X[:5])
    assert np.allclose(a[:5], b, atol=1e-6)
    assert np.allclose(a.mean(axis=1), 0, atol=1e-4) and np.allclose(a.std(axis=1), 1, atol=1e-3)


def test_windows_are_slices_of_stored_360():
    X = np.arange(360, dtype=np.float32)[None, :]
    w216, w280, w360 = crop(X, CFG, 216), crop(X, CFG, 280), crop(X, CFG, 360)
    assert w216.shape == (1, 216) and w216[0, 0] == 90 and w216[0, 90] == 180   # R à l'indice 180 de la fenêtre stockée
    assert w280.shape == (1, 280) and w280[0, 0] == 70 and w280[0, 110] == 180
    assert w360.shape == (1, 360) and w360[0, 180] == 180


@needs_cache
def test_prepared_has_no_leak_and_r_peak_is_centred():
    p = prepare("P2", 216, CFG)
    assert p.pre.fitted_on == "train" and p.info["window"] == 216
    d = np.load(dp.cache_path(CFG, "train"))
    idx = np.flatnonzero(d["y"] == 0)[:2000]
    win = crop(d["X_bp"][idx], CFG, 216)
    peak = np.abs(win - np.median(win, axis=1, keepdims=True))[:, 60:120].argmax(axis=1) + 60
    assert np.mean(np.abs(peak - 90) <= 12) > 0.9


# ---------- augmentation ----------

def test_augmentation_only_on_train():
    x = torch.randn(8, 1, 216)
    ok = ECGData(np.zeros((8, 216)), np.zeros((8, 5)), np.zeros(8), "train")
    assert ok.augment_batch(x).shape == x.shape and ok.n_augmented_batches == 1
    for s in ("val", "test"):
        d = ECGData(np.zeros((8, 216)), np.zeros((8, 5)), np.zeros(8), s)
        with pytest.raises(PermissionError):
            d.augment_batch(x)
        with pytest.raises(PermissionError):
            d.loader(4, weighted=True)


# ---------- pertes ----------

def test_focal_gamma0_equals_bce_and_pos_weight():
    z = torch.randn(64)
    y = (torch.rand(64) > 0.7).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    assert FocalLoss(gamma=0.0)(z, y).item() == pytest.approx(bce.item(), rel=1e-5)
    assert FocalLoss(gamma=2.0)(z, y).item() < bce.item()
    ytr = np.array([0] * 75 + [1] * 25)
    assert positive_weight(ytr) == pytest.approx(3.0)
    assert float(build_loss({"name": "bce_pw"}, ytr).pos_weight) == pytest.approx(3.0)


# ---------- modèles ----------

@pytest.mark.parametrize("arch", sorted(PRESETS))
@pytest.mark.parametrize("n_rr", [0, 5])
@pytest.mark.parametrize("length", [216, 280])
def test_model_forward_shapes(arch, n_rr, length):
    m = build_model(arch, length=length, n_rr=n_rr).eval()
    x, rr = torch.randn(4, 1, length), torch.randn(4, 5) if n_rr else None
    assert m(x, rr).shape == (4,)


@pytest.mark.parametrize("kw", [{"kernel": "multi"}, {"pool": "stride"}, {"head": "flatten"}, {"bn": False, "act": "gelu"}])
def test_cnn_variants(kw):
    assert build_model("cnn_m", kw, 216, 0)(torch.randn(2, 1, 216)).shape == (2,)


@pytest.mark.parametrize("kw", [{"pooling": "mean"}, {"pooling": "attention"}, {"layers": 2, "rnn_dropout": 0.2},
                                {"kernel": "multi"}, {"bidirectional": True}])
def test_cnnlstm_variants_and_sequence_length(kw):
    m = build_model("cnnlstm_s", kw, 216, 5)
    assert m.seq_len == 54
    assert m(torch.randn(2, 1, 216), torch.randn(2, 5)).shape == (2,)


def test_cnn_cam_shape():
    m = build_model("cnn_m", {"head": "gap"}, 216, 0).eval()
    assert m.cam(torch.randn(3, 1, 216)).shape == (3, 216)


def test_eval_mode_is_deterministic():
    torch.manual_seed(0)
    m = build_model("cnnlstm_s", {"cnn_dropout": 0.5}, 216, 0).eval()
    x = torch.randn(4, 1, 216)
    assert torch.equal(m(x), m(x))
