"""Préparation des entrées : recadrage de fenêtre, normalisations P0-P3 (ajustées sur train uniquement), conteneur de données."""
import json
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from . import augmentation
from .data_processing import SPLIT_NAMES, SPLITS_PATH, assert_split_integrity, cache_path, load_config
from .features import feature_matrix, rr_matrix

def window_geometry(cfg, window):
    pre, post = cfg["windows"][str(window)]
    return pre, post


def crop(X, cfg, window):
    pre, post = window_geometry(cfg, window)
    start = cfg["windows"]["stored"][0] - pre
    return X[:, start:start + pre + post]


class Preprocessor:
    """P0..P3. Les statistiques globales ne sont ajustées que sur le split 'train'."""

    def __init__(self, mode):
        assert mode in ("P0", "P1", "P2", "P3"), mode
        self.mode = mode
        self.fitted_on = None
        self.stats = {}

    def source(self, X_raw, X_bp):
        return X_raw if self.mode == "P0" else X_bp

    def fit(self, X_raw, X_bp, split):
        if split != "train":
            raise ValueError(f"Preprocessor.fit autorisé sur 'train' uniquement, reçu '{split}'")
        X = self.source(X_raw, X_bp)
        if self.mode == "P2":
            self.stats = {"mean": float(X.mean()), "std": float(X.std() + 1e-8)}
        elif self.mode == "P3":
            q1, med, q3 = np.percentile(X, [25, 50, 75])
            self.stats = {"median": float(med), "iqr": float(q3 - q1 + 1e-8)}
        self.fitted_on = split
        return self

    def transform(self, X_raw, X_bp):
        assert self.fitted_on == "train", "transformer avant fit(train)"
        X = self.source(X_raw, X_bp)
        if self.mode in ("P0", "P1"):
            return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
        if self.mode == "P2":
            return (X - self.stats["mean"]) / self.stats["std"]
        return (X - self.stats["median"]) / self.stats["iqr"]


@dataclass
class Prepared:
    preprocess: str
    window: int
    centre: int
    X: dict = field(default_factory=dict)       # [n, L] normalisé
    RR: dict = field(default_factory=dict)      # [n, 5] standardisé (stats train)
    F: dict = field(default_factory=dict)       # [n, 14] features LR/RF (brutes)
    y: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    pre: Preprocessor = None
    rr_stats: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


_CACHE = {}


def prepare(preprocess="P1", window=360, cfg=None):
    cfg = cfg or load_config()
    key = (preprocess, window)
    if key in _CACHE:
        return _CACHE[key]

    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    assert_split_integrity(splits, cfg)

    raw = {}
    for s in SPLIT_NAMES:
        with np.load(cache_path(cfg, s)) as d:
            raw[s] = {k: d[k] for k in d.files}
    centre = window_geometry(cfg, window)[0]

    pre = Preprocessor(preprocess)
    pre.fit(crop(raw["train"]["X_raw"], cfg, window), crop(raw["train"]["X_bp"], cfg, window), "train")
    rr_cols = cfg["defaults"]["deep"].get("rr_cols")  # sous-ensemble des 5 colonnes RR pour les réseaux (None = les 5)
    rr_train = rr_matrix(raw["train"])
    rr_mean, rr_std = rr_train.mean(axis=0), rr_train.std(axis=0) + 1e-8

    p = Prepared(preprocess, window, centre, pre=pre,
                 rr_stats={"mean": rr_mean.tolist(), "std": rr_std.tolist()})
    for s in SPLIT_NAMES:
        r = raw[s]
        Xr, Xb = crop(r["X_raw"], cfg, window), crop(r["X_bp"], cfg, window)
        rr = rr_matrix(r)
        p.X[s] = pre.transform(Xr, Xb).astype(np.float32)
        p.RR[s] = ((rr - rr_mean) / rr_std).astype(np.float32)
        if rr_cols is not None:
            p.RR[s] = np.ascontiguousarray(p.RR[s][:, rr_cols])
        p.F[s] = feature_matrix(Xb, rr, centre)
        p.y[s] = r["y"]
        p.meta[s] = {k: r[k] for k in ("record_id", "group_id", "symbol", "sample", "split")}
    p.info = {
        "preprocess": preprocess, "window": window, "centre": centre,
        "fitted_on": pre.fitted_on, "stats": pre.stats, "rr_stats": p.rr_stats,
        "n": {s: int(len(p.y[s])) for s in SPLIT_NAMES},
        "prevalence": {s: float(p.y[s].mean()) for s in SPLIT_NAMES},
    }
    assert_prepared(p, cfg)
    _CACHE[key] = p
    return p


def assert_prepared(p, cfg):
    """Contrôles anti-fuite exécutés à chaque préparation."""
    assert p.pre.fitted_on == "train"
    for s in SPLIT_NAMES:
        assert (p.meta[s]["split"] == s).all(), f"étiquette de split incohérente dans {s}"
        assert np.isfinite(p.X[s]).all() and np.isfinite(p.F[s]).all()
    recs = {s: set(p.meta[s]["record_id"].tolist()) for s in SPLIT_NAMES}
    grps = {s: set(p.meta[s]["group_id"].tolist()) for s in SPLIT_NAMES}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert recs[a].isdisjoint(recs[b]) and grps[a].isdisjoint(grps[b]), f"fuite de patient {a}/{b}"
    for g in cfg["records"]["same_patient_groups"]:
        assert len({s for s in SPLIT_NAMES if recs[s] & set(g)}) <= 1


class ECGData:
    """Un split sous forme de tenseurs. L'augmentation n'est possible que sur le split 'train'."""

    def __init__(self, X, RR, y, split):
        self.split = split
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).unsqueeze(1)
        self.RR = torch.from_numpy(np.ascontiguousarray(RR, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))
        self.n_augmented_batches = 0

    def __len__(self):
        return len(self.y)

    def augment_batch(self, xb, aug_cfg=None):
        if self.split != "train":
            raise PermissionError(f"augmentation interdite sur le split '{self.split}'")
        self.n_augmented_batches += 1
        return augmentation.augment(xb, **(aug_cfg or {}))

    def loader(self, batch_size, shuffle=False, weighted=False, seed=0):
        ds = TensorDataset(self.X, self.RR, self.y)
        g = torch.Generator().manual_seed(seed)
        if weighted:
            if self.split != "train":
                raise PermissionError("échantillonnage pondéré réservé à 'train'")
            freq = torch.bincount(self.y.long()).float()
            w = (1.0 / freq)[self.y.long()]
            sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True, generator=g)
            return DataLoader(ds, batch_size=batch_size, sampler=sampler)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None)


def make_data(p):
    return {s: ECGData(p.X[s], p.RR[s], p.y[s], s) for s in SPLIT_NAMES}
