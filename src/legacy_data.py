"""Chargement des .npz produits par scripts/preprocess.py (protocole de Meriem) et features."""
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FS = 360

RR_NAMES = ["rr_pre", "rr_post", "rr_ratio"]
MORPH_NAMES = [
    "max", "min", "range", "energy", "area", "std",
    "max_slope", "qrs_width_s", "r_amplitude",
]
FEATURE_NAMES = RR_NAMES + MORPH_NAMES


def load_split(name, data_dir=DATA_DIR):
    path = Path(data_dir) / f"mitdb_{name}.npz"
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def morphology_features(X, fs=FS):
    """Features de forme d'onde calculées sur chaque fenêtre (R centré à l'indice L//2)."""
    c = X.shape[1] // 2
    dx = np.diff(X, axis=1)
    centre = X[:, c - 40:c + 40]
    dev = np.abs(centre - np.median(X, axis=1, keepdims=True))
    qrs_width = (dev > 0.5 * dev.max(axis=1, keepdims=True)).sum(axis=1) / fs
    return np.column_stack([
        X.max(axis=1),
        X.min(axis=1),
        X.max(axis=1) - X.min(axis=1),
        (X ** 2).sum(axis=1),
        np.abs(X).sum(axis=1),
        X.std(axis=1),
        np.abs(dx).max(axis=1),
        qrs_width,
        X[:, c],
    ]).astype(np.float32)


def feature_matrix(X, RR):
    return np.hstack([RR, morphology_features(X)])


def describe(splits):
    for name, d in splits.items():
        y = d["y"]
        print(
            f"{name:5s} n={len(y):6d} normal={int((y == 0).sum()):6d} "
            f"anormal={int((y == 1).sum()):5d} ({100 * y.mean():.1f} %) "
            f"patients={len(set(d['patient_id']))}"
        )
    seen = {}
    for name, d in splits.items():
        for pid in set(d["patient_id"]):
            assert pid not in seen, f"patient {pid} dans {seen[pid]} et {name}"
            seen[pid] = name
