"""Features RR et morphologie pour LR/RF (calculées sur la fenêtre filtrée, avant toute normalisation par battement)."""
import numpy as np

FS = 360
RR_NAMES = ["rr_prev", "rr_next", "rr_med", "rr_ratio_prev", "rr_ratio_next"]
MORPH_NAMES = [
    "max", "min", "range", "energy", "area", "std",
    "max_slope", "qrs_width_s", "r_amplitude",
]
FEATURE_NAMES = RR_NAMES + MORPH_NAMES


def morphology_features(X, centre, fs=FS, qrs_half=40):
    """X: [n, L] en mV ; centre: indice de l'annotation R dans la fenêtre."""
    dx = np.diff(X, axis=1)
    seg = X[:, centre - qrs_half:centre + qrs_half]
    dev = np.abs(seg - np.median(X, axis=1, keepdims=True))
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
        X[:, centre],
    ]).astype(np.float32)


def rr_matrix(meta):
    return np.column_stack([meta[k] for k in
                            ("rr_prev", "rr_next", "rr_med", "rr_ratio_prev", "rr_ratio_next")]).astype(np.float32)


def feature_matrix(X_bp, rr_raw, centre):
    return np.hstack([rr_raw, morphology_features(X_bp, centre)])
