"""Pipeline WFDB -> fenêtres de battements. Le split par patient est décidé AVANT l'extraction."""
import itertools
import json
import os
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / os.environ.get("EXPERIMENT_CONFIG", "configs/experiment.json")
SPLITS_PATH = ROOT / "configs" / "splits.json"
SPLIT_NAMES = ("train", "val", "test")


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def raw_dir(cfg):
    return ROOT / cfg["dataset"]["raw_dir"]


def cache_dir(cfg):
    return ROOT / cfg["dataset"]["cache_dir"]


def cache_path(cfg, split):
    return cache_dir(cfg) / f"mitdb_{cfg['label_mapping']['version']}_{split}.npz"


# ---------- étiquettes ----------

def label_of(symbol, cfg):
    """0 normal, 1 anormal, None si exclu ou non-battement."""
    lm = cfg["label_mapping"]
    if symbol in lm["normal"]:
        return 0
    if symbol in lm["abnormal"]:
        return 1
    return None


def beat_symbols(cfg):
    lm = cfg["label_mapping"]
    return set(lm["normal"]) | set(lm["abnormal"]) | set(lm["excluded_beats"])


# ---------- groupes patients et split ----------

def group_of(rec, cfg):
    for g in cfg["records"]["same_patient_groups"]:
        if rec in g:
            return "_".join(sorted(g))
    return rec


def record_counts(rec, cfg):
    ann = wfdb.rdann(str(raw_dir(cfg) / rec), "atr")
    syms = [s for s in ann.symbol if s in beat_symbols(cfg)]
    labels = [label_of(s, cfg) for s in syms]
    abn = {}
    for s, x in zip(syms, labels):
        if x == 1:
            abn[s] = abn.get(s, 0) + 1
    return {
        "normal": sum(1 for x in labels if x == 0),
        "abnormal": sum(1 for x in labels if x == 1),
        "excluded": sum(1 for x in labels if x is None),
        "abnormal_symbols": abn,
    }


def make_splits(cfg, force=False):
    """Split par groupes de patients, figé dans configs/splits.json. Critère : comptes d'annotations uniquement."""
    if SPLITS_PATH.exists() and not force:
        return json.loads(SPLITS_PATH.read_text(encoding="utf-8"))

    rc = cfg["records"]
    test_groups = {group_of(r, cfg) for r in rc["ds2"]}
    test_records = list(rc["ds2"]) + [r for r in rc["ds1"] if group_of(r, cfg) in test_groups and r not in rc["ds2"]]
    dev_records = [r for r in rc["ds1"] if group_of(r, cfg) not in test_groups]
    counts = {r: record_counts(r, cfg) for r in dev_records + test_records}

    members = {}
    for r in dev_records:
        members.setdefault(group_of(r, cfg), []).append(r)
    groups = sorted(members)
    sp = cfg["split"]

    def prevalence(gs):
        a = sum(counts[r]["abnormal"] for g in gs for r in members[g])
        n = sum(counts[r]["normal"] for g in gs for r in members[g])
        return a, a / max(a + n, 1)

    def type_dist(gs):
        """Distribution des types de battements anormaux (L, R, V, A, F, autres)."""
        tot = {}
        for g in gs:
            for r in members[g]:
                for s, c in counts[r]["abnormal_symbols"].items():
                    k = s if s in "LRVAF" else "other"
                    tot[k] = tot.get(k, 0) + c
        z = max(sum(tot.values()), 1)
        return np.array([tot.get(k, 0) / z for k in ("L", "R", "V", "A", "F", "other")])

    def max_record_share(gs):
        a = [counts[r]["abnormal"] for g in gs for r in members[g]]
        return max(a) / max(sum(a), 1)

    best = None
    for val_g in itertools.combinations(groups, sp["n_val_groups"]):
        tr_g = [g for g in groups if g not in val_g]
        a_val, p_val = prevalence(val_g)
        _, p_tr = prevalence(tr_g)
        if a_val < sp["min_abnormal_val"] or max_record_share(val_g) > sp["max_record_share_val"]:
            continue
        tv = 0.5 * np.abs(type_dist(val_g) - type_dist(tr_g)).sum()
        score = abs(p_val - p_tr) + sp["type_weight"] * tv
        if best is None or score < best[0]:
            best = (score, val_g, tr_g)
    assert best is not None, "aucun split de validation ne satisfait les contraintes"
    score, val_g, tr_g = best

    splits = {
        "label_mapping_version": cfg["label_mapping"]["version"],
        "selection": {"search": "exhaustive over group combinations", "score": float(score), "uses": "annotation counts only"},
        "train": sorted(r for g in tr_g for r in members[g]),
        "val": sorted(r for g in val_g for r in members[g]),
        "test": sorted(test_records),
        "record_counts": counts,
    }
    assert_split_integrity(splits, cfg)
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_PATH.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return splits


def assert_split_integrity(splits, cfg):
    recs = {s: set(splits[s]) for s in SPLIT_NAMES}
    for a in SPLIT_NAMES:
        for b in SPLIT_NAMES:
            if a < b:
                assert recs[a].isdisjoint(recs[b]), f"enregistrements communs {a}/{b}: {recs[a] & recs[b]}"
    grp = {s: {group_of(r, cfg) for r in recs[s]} for s in SPLIT_NAMES}
    for a in SPLIT_NAMES:
        for b in SPLIT_NAMES:
            if a < b:
                assert grp[a].isdisjoint(grp[b]), f"patient commun {a}/{b}: {grp[a] & grp[b]}"
    for g in cfg["records"]["same_patient_groups"]:
        where = {s for s in SPLIT_NAMES if recs[s] & set(g)}
        assert len(where) <= 1, f"enregistrements {g} séparés entre {where}"
    used = set().union(*recs.values())
    assert used.isdisjoint(cfg["records"]["paced_excluded"]), "enregistrement à stimulateur présent"


# ---------- extraction ----------

def bandpass(sig, fs, bp):
    sos = butter(bp["order"], [bp["low_hz"], bp["high_hz"]], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)


def extract_record(rec, cfg):
    """Fenêtres brutes et filtrées de 360 échantillons pour tous les battements étiquetables d'un enregistrement."""
    pre, post = cfg["windows"]["stored"]
    record = wfdb.rdrecord(str(raw_dir(cfg) / rec))
    ann = wfdb.rdann(str(raw_dir(cfg) / rec), "atr")
    lead = cfg["dataset"]["lead"]
    assert lead in record.sig_name, f"{rec}: dérivation {lead} absente ({record.sig_name})"
    raw = record.p_signal[:, record.sig_name.index(lead)].astype(np.float64)
    bp = bandpass(raw, record.fs, cfg["bandpass"])

    bs = beat_symbols(cfg)
    idx = [i for i, s in enumerate(ann.symbol) if s in bs]
    samples = ann.sample[idx].astype(np.int64)
    symbols = [ann.symbol[i] for i in idx]
    rr = np.diff(samples) / record.fs

    out = {k: [] for k in ("X_raw", "X_bp", "y", "symbol", "sample", "rr_prev", "rr_next", "rr_med")}
    for i in range(1, len(samples) - 1):
        y = label_of(symbols[i], cfg)
        r = samples[i]
        if y is None or r - pre < 0 or r + post > len(raw):
            continue
        out["X_raw"].append(raw[r - pre:r + post])
        out["X_bp"].append(bp[r - pre:r + post])
        out["y"].append(y)
        out["symbol"].append(symbols[i])
        out["sample"].append(r)
        out["rr_prev"].append(rr[i - 1])
        out["rr_next"].append(rr[i])
        out["rr_med"].append(np.median(rr[max(0, i - 8):i]))
    n = len(out["y"])
    return {
        "X_raw": np.asarray(out["X_raw"], np.float32).reshape(n, pre + post),
        "X_bp": np.asarray(out["X_bp"], np.float32).reshape(n, pre + post),
        "y": np.asarray(out["y"], np.int64),
        "symbol": np.asarray(out["symbol"], dtype="<U2"),
        "sample": np.asarray(out["sample"], np.int64),
        "rr_prev": np.asarray(out["rr_prev"], np.float32),
        "rr_next": np.asarray(out["rr_next"], np.float32),
        "rr_med": np.asarray(out["rr_med"], np.float32),
        "record_id": np.full(n, rec, dtype="<U3"),
        "group_id": np.full(n, group_of(rec, cfg), dtype="<U8"),
        "lead": np.full(n, lead, dtype="<U8"),
    }


def build_cache(cfg, force=False):
    splits = make_splits(cfg, force=False)
    assert_split_integrity(splits, cfg)
    cache_dir(cfg).mkdir(parents=True, exist_ok=True)
    manifest = {"label_mapping": cfg["label_mapping"], "splits": {}}
    for split in SPLIT_NAMES:
        path = cache_path(cfg, split)
        if path.exists() and not force:
            with np.load(path) as d:
                y, rid = d["y"], d["record_id"]
        else:
            parts = [extract_record(r, cfg) for r in splits[split]]
            data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            data["rr_ratio_prev"] = data["rr_prev"] / np.maximum(data["rr_med"], 1e-6)
            data["rr_ratio_next"] = data["rr_next"] / np.maximum(data["rr_med"], 1e-6)
            data["split"] = np.full(len(data["y"]), split, dtype="<U5")
            np.savez_compressed(path, **data)
            y, rid = data["y"], data["record_id"]
        manifest["splits"][split] = {
            "records": splits[split],
            "n_beats": int(len(y)),
            "n_normal": int((y == 0).sum()),
            "n_abnormal": int((y == 1).sum()),
            "per_record": {r: [int((rid == r).sum()), int(y[rid == r].sum())] for r in splits[split]},
        }
    (cache_dir(cfg) / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
