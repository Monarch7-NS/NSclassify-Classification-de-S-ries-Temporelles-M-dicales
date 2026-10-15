"""Traçabilité : identifiants d'expérience, métadonnées d'exécution, index machine-lisible des résultats."""
import csv
import hashlib
import io
import json
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / os.environ.get("RESULTS_DIR", "results") / "experiments"
INDEX_JSON = EXP_DIR / "index.json"
INDEX_CSV = EXP_DIR / "index.csv"
EXCLUDE_FROM_ID = ("factor", "notes", "id")


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_id(cfg):
    core = {k: v for k, v in cfg.items() if k not in EXCLUDE_FROM_ID}
    h = hashlib.sha1(json.dumps(core, sort_keys=True).encode()).hexdigest()[:8]
    return f"{cfg.get('stage', 'adhoc')}-{cfg['model']['family']}-{h}"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def run_info(cfg, device, prep_info, splits_hash):
    import scipy
    import sklearn
    import torch
    return {
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
        "device": str(device),
        "seed": cfg.get("seed"),
        "splits_sha1": splits_hash,
        "preprocessing": prep_info,
    }


def file_sha1(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


@contextmanager
def _index_lock(timeout=60):
    lock = EXP_DIR / "index.lock"
    t0 = time.time()
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            if time.time() - t0 > timeout:  # verrou orphelin d'un processus tué
                lock.unlink(missing_ok=True)
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _atomic_write(path, text):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    for _ in range(100):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # lecteur concurrent sous Windows
            time.sleep(0.05)
    os.replace(tmp, path)


def load_index():
    for _ in range(100):
        if not INDEX_JSON.exists():
            return {}
        try:
            return json.loads(INDEX_JSON.read_text(encoding="utf-8"))
        except (PermissionError, json.JSONDecodeError):
            time.sleep(0.05)
    return json.loads(INDEX_JSON.read_text(encoding="utf-8"))


def update_index(exp_id, row):
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    with _index_lock():
        idx = load_index()
        idx[exp_id] = row
        _atomic_write(INDEX_JSON, json.dumps(idx, indent=1, default=str))
        cols = sorted({k for r in idx.values() for k in r})
        first = [c for c in ("id", "stage", "family", "arch", "factor", "status") if c in cols]
        cols = first + [c for c in cols if c not in first]
        buf = io.StringIO(newline="")
        w = csv.DictWriter(buf, fieldnames=cols)
        w.writeheader()
        for r in idx.values():
            w.writerow({k: r.get(k, "") for k in cols})
        _atomic_write(INDEX_CSV, buf.getvalue())


def flat_row(exp_id, cfg, result):
    """Ligne d'index plate : configuration + métriques validation (+ test si présent) + coût."""
    m = cfg["model"]
    opt = cfg.get("optimizer", {})
    row = {
        "id": exp_id, "stage": cfg.get("stage"), "family": m["family"], "arch": m.get("arch", ""),
        "factor": cfg.get("factor", ""), "status": result.get("status"),
        "preprocess": cfg.get("preprocess"), "window": cfg.get("window"), "use_rr": cfg.get("use_rr"),
        "loss": json.dumps(cfg.get("loss")), "optimizer": opt.get("name"), "lr": opt.get("lr"),
        "weight_decay": opt.get("weight_decay"), "batch_size": cfg.get("batch_size"),
        "scheduler": cfg.get("scheduler"), "augment": cfg.get("augment"), "sampler": cfg.get("sampler"),
        "params_json": json.dumps(m.get("params", {})), "seed": cfg.get("seed"),
    }
    for k in ("best_epoch", "epochs_run", "train_time_s", "epoch_time_s", "n_params", "model_bytes",
              "latency_ms_single", "throughput_beats_s", "best_params", "note"):
        if k in result:
            row[k] = result[k]
    for split in ("val", "test"):
        for thr_name, mm in result.get(f"{split}_metrics", {}).items():
            if thr_name != result.get("primary_threshold", "max_f1") and not thr_name.startswith("__"):
                continue
            for k in ("pr_auc", "roc_auc", "f1", "sensitivity", "specificity", "precision", "mcc",
                      "balanced_accuracy", "fp", "fn", "threshold"):
                row[f"{split}_{k}"] = mm[k]
    for k in ("overfit_gap_last_epoch", "augmented_batches_train", "augmented_batches_val_test"):
        if k in result:
            row[k] = result[k]
    return row
