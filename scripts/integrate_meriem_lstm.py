"""Intègre les 3 checkpoints du LSTM de Meriem (results/meriem_lstm/best_lstm_seed_*.pth) au suivi d'expériences du projet.

Le code d'entraînement de Meriem n'est pas modifié : on relit ses checkpoints avec sa classe PureLSTMClassifier, on recalcule
les probabilités val/test et on passe par les mêmes fonctions d'évaluation que les autres modèles (src.train.evaluate_run).
Entrées créées : 1 run `baseline` (seed 42, validation uniquement) + 3 runs `final` (seeds 42, 43, 44, test avec seuil figé sur val).
"""
import ast
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import evaluate as ev  # noqa: E402
from src import experiment_tracker as tr  # noqa: E402
from src.data_processing import SPLITS_PATH  # noqa: E402
from src.datasets import prepare  # noqa: E402
from src.models import count_params  # noqa: E402
from src.train import evaluate_run, predict_logits, sigmoid  # noqa: E402
from scripts.run_meriem_lstm import ORIGINAL, RR_COLS, SANDBOX  # noqa: E402

SEEDS = (42, 43, 44)
NOTE = ("LSTM de Meriem, code inchange (scripts/meriem_lstm_original.py) ; entrees P1 fenetre 360 + 3 RR "
        f"(colonnes {RR_COLS} de rr_matrix) ; checkpoints results/meriem_lstm/")


def her_model_class():
    """Extrait uniquement PureLSTMClassifier du script de Meriem (sans exécuter son entraînement)."""
    tree = ast.parse(ORIGINAL.read_text(encoding="utf-8"))
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
            or (isinstance(n, ast.ClassDef) and n.name == "PureLSTMClassifier")]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(ORIGINAL), "exec"), ns)
    return ns["PureLSTMClassifier"]


class Adapter(nn.Module):
    """Interface du projet forward(x[B,1,L], rr[B,5]) -> logits[B] autour du modèle de Meriem (x[B,L,1], rr[B,3])."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.register_buffer("cols", torch.tensor(RR_COLS))

    def forward(self, x, rr):
        return self.inner(x.transpose(1, 2), rr if rr.shape[1] == 3 else rr[:, self.cols]).squeeze(-1)


def parse_log():
    """Découpe train.log par seed : historique par époque, temps, valeurs rapportées par son script."""
    t_of = re.compile(r"^\[\s*([\d.]+)s\] (.*)$")
    ep = re.compile(r"Époque \[(\d+)/\d+\] - Train Loss: ([\d.]+) - Val Loss: ([\d.]+) - Val PR-AUC: ([\d.]+)")
    blocks, cur = {}, None
    for line in (SANDBOX / "train.log").read_text(encoding="utf-8").splitlines():
        m = t_of.match(line)
        if not m:
            continue
        t, txt = float(m.group(1)), m.group(2)
        h = re.search(r"ENTRAINEMENT - SEED (\d+)", txt)
        if h:
            cur = blocks.setdefault(int(h.group(1)), {"t0": t, "epochs": [], "reported": {}})
            continue
        if cur is None:
            continue
        e = ep.search(txt)
        if e:
            cur["epochs"].append({"epoch": int(e.group(1)), "t": t, "train_loss": float(e.group(2)),
                                  "val_loss": float(e.group(3)), "val_pr_auc": float(e.group(4))})
        for key, pat in (("best_epoch", r"Meilleure époque\s*:\s*(\d+)"), ("threshold", r"Seuil choisi\s*:\s*([\d.]+)"),
                         ("test_pr_auc", r"Test PR-AUC\s*:\s*([\d.]+)"), ("test_f1", r"Test F1\s*:\s*([\d.]+)")):
            r = re.search(pat, txt)
            if r and key not in cur["reported"]:
                cur["reported"][key] = float(r.group(1))
    return blocks


def history_frame(block):
    rows, prev = [], block["t0"]
    for e in block["epochs"]:
        rows.append({"epoch": e["epoch"], "train_loss": e["train_loss"], "val_loss": e["val_loss"], "val_bce": np.nan,
                     "val_pr_auc": e["val_pr_auc"], "val_roc_auc": np.nan, "epoch_time_s": e["t"] - prev})
        prev = e["t"]
    return pd.DataFrame(rows)


def lstm_config(stage, seed):
    return {
        "preprocess": "P1", "window": 360, "use_rr": True, "loss": {"name": "bce_pw"},
        "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4}, "scheduler": "none", "batch_size": 128,
        "epochs": 30, "patience": 5, "clip_grad": 1.0, "augment": False, "sampler": "none",
        "model": {"family": "lstm", "arch": "meriem_pure_lstm", "params": {"hidden": 128, "layers": 2}},
        "stage": stage, "seed": seed, "factor": "final<-meriem_original" if stage == "final" else "meriem_original",
    }


def integrate_one(stage, seed, prep, cls, block):
    cfg = lstm_config(stage, seed)
    exp_id = tr.config_id(cfg)
    run_dir = tr.EXP_DIR / exp_id
    done = run_dir / "metrics.json"
    if done.exists() and json.loads(done.read_text(encoding="utf-8")).get("status") == "ok":
        print("deja integre :", exp_id)
        return exp_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = SANDBOX / f"best_lstm_seed_{seed}.pth"
    inner = cls()
    inner.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = Adapter(inner).eval()
    shutil.copy(ckpt, run_dir / "best.pt")
    hist = history_frame(block)
    hist.to_csv(run_dir / "history.csv", index=False)

    def predict_arrays(X, RR):
        return sigmoid(predict_logits(model, X, RR, True, "cpu"))

    fit = {"model": model, "predict_arrays": predict_arrays, "kind": "dl", "use_rr": True,
           "predict": lambda s: predict_arrays(prep.X[s], prep.RR[s])}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    result = evaluate_run(cfg, prep, fit, run_dir, "cpu", final=(stage == "final"))

    x1 = torch.from_numpy(prep.X["val"][:1]).unsqueeze(1)
    r1 = torch.from_numpy(prep.RR["val"][:1])
    xb = torch.from_numpy(prep.X["val"][:1024]).unsqueeze(1)
    rb = torch.from_numpy(prep.RR["val"][:1024])
    with torch.no_grad():
        latency = 1000 * ev.time_callable(lambda: model(x1, r1), repeats=100)
        throughput = len(xb) / ev.time_callable(lambda: model(xb, rb), repeats=5, warmup=1)
    best_epoch = int(block["reported"]["best_epoch"])
    result.update({
        "best_epoch": best_epoch, "epochs_run": len(hist), "n_params": count_params(model),
        "train_time_s": float(block["epochs"][-1]["t"] - block["t0"]), "epoch_time_s": float(hist.epoch_time_s.mean()),
        "model_bytes": ckpt.stat().st_size, "latency_ms_single": latency, "throughput_beats_s": throughput,
        "status": "ok", "note": NOTE, "reported_by_original_script": block["reported"],
    })
    (run_dir / "run_info.json").write_text(
        json.dumps(tr.run_info(cfg, "cpu", prep.info, tr.file_sha1(SPLITS_PATH)), indent=2, default=str), encoding="utf-8")
    result["id"] = exp_id
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    tr.update_index(exp_id, tr.flat_row(exp_id, cfg, result))

    thr_mine = result["thresholds_from_val"]["max_f1"]
    rep = block["reported"]
    line = f"{exp_id} seed={seed} val PR-AUC={result['val_metrics']['max_f1']['pr_auc']:.4f} thr={thr_mine:.4f}"
    if stage == "final":
        line += (f" | test PR-AUC={result['test_metrics']['max_f1']['pr_auc']:.4f} "
                 f"(son script : {rep.get('test_pr_auc')}, seuil {rep.get('threshold')})")
    print(line, flush=True)
    return exp_id


def integrate():
    blocks = parse_log()
    missing = [s for s in SEEDS if s not in blocks or not (SANDBOX / f"best_lstm_seed_{s}.pth").exists()
               or "test_pr_auc" not in blocks[s]["reported"]]
    if missing:
        raise SystemExit(f"entrainement incomplet pour les seeds {missing}")
    prep = prepare("P1", 360)
    cls = her_model_class()
    integrate_one("baseline", 42, prep, cls, blocks[42])
    for s in SEEDS:
        integrate_one("final", s, prep, cls, blocks[s])


if __name__ == "__main__":
    integrate()
