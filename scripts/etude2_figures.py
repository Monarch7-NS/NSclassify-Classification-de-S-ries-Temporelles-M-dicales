"""Figures de l'étude 2 -> results_etude2/figures/ (uniquement à partir de fichiers produits).

Usage : python -m scripts.etude2_figures
"""
import ast
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_presentation as bp  # noqa: E402

E2 = ROOT / "results_etude2"
OUT = E2 / "figures"
FAMS = ["lr", "rf", "mlp", "cnn", "cnn_lstm"]
NAME = {"lr": "LR", "rf": "RF", "mlp": "MLP", "cnn": "CNN 1D", "cnn_lstm": "CNN-LSTM"}
COL = {"lr": "#8d99ae", "rf": "#6d7586", "mlp": "#1f8a8a", "cnn": "#e4572e", "cnn_lstm": "#e9a23b"}


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)


def finals():
    out = {}
    for f in FAMS:
        out[f] = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((E2 / "final").glob(f"{f}__seed*.json"))]
    return out


def fig_cv_folds():
    fig, axes = plt.subplots(1, len(FAMS), figsize=(15, 4.2), sharey=True)
    for ax, f in zip(axes, FAMS):
        rows = [r for r in csv.DictReader((E2 / f"search_table_{f}.csv").open(encoding="utf-8")) if r["folds_done"] == "5"]
        means = [float(r["cv_pr_auc_mean"]) for r in rows]
        best = int(np.argmax(means))
        for i, r in enumerate(rows):
            v = ast.literal_eval(r["fold_pr_auc"])
            ax.scatter([i] * len(v), v, s=22, color=COL[f], alpha=.55)
            ax.hlines(means[i], i - .3, i + .3, color="k", lw=2 if i == best else 1)
        ax.set_title(f"{NAME[f]} ({len(rows)} réglages)")
        ax.set_xticks(range(len(rows)))
        ax.set_xlabel("réglage")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("PR-AUC par pli (5 plis groupés)")
    fig.suptitle("CV : l'écart entre plis dépasse l'écart entre réglages (trait = moyenne, épais = réglage choisi)")
    save(fig, "E2_01_cv_folds")


def fig_test_seeds(fin):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for i, f in enumerate(FAMS):
        v = [j["test_at_oof_thr"]["pr_auc"] for j in fin[f]]
        ax.scatter(np.full(len(v), i) + np.linspace(-.12, .12, len(v)), v, color=COL[f], s=30)
        ax.hlines(np.mean(v), i - .3, i + .3, color="k", lw=2)
    ax.axhline(bp.bbb_records()[2] / bp.bbb_records()[1], color="gray", ls="--", label="hasard (prévalence)")
    ax.set_xticks(range(len(FAMS)), [NAME[f] for f in FAMS])
    ax.set_ylabel("PR-AUC test (une valeur par seed)")
    ax.set_title("Étude 2 : PR-AUC test finale")
    ax.legend()
    ax.grid(alpha=.3)
    save(fig, "E2_02_test_seeds")


def fig_pr_curves(fin):
    z = np.load(ROOT / "data" / "processed" / "mitdb_strict_v1_test.npz", allow_pickle=True)
    y = z["y"].astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    for f in FAMS:
        probs = [np.load(p) for p in sorted((E2 / "final").glob(f"{f}__seed*_test_probs.npy"))]
        p = np.mean(probs, axis=0).ravel()
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, color=COL[f], label=f"{NAME[f]} (AP {average_precision_score(y, p):.3f})")
    ax.axhline(y.mean(), color="gray", ls="--", label=f"hasard ({y.mean():.3f})")
    ax.set_xlabel("rappel")
    ax.set_ylabel("précision")
    ax.set_title("Test : précision-rappel de la moyenne des probabilités des seeds", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    save(fig, "E2_03_pr_curves")


def fig_study1_vs_2(fin):
    s1 = {r["k"]: r for r in bp.study1()}
    x = np.arange(len(FAMS))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for off, lab, col, get in [(-.2, "étude 1", "#c9c1b2", lambda f: (s1[f]["pr"], s1[f]["sd"])),
                               (.2, "étude 2", "#1f8a8a", lambda f: (np.mean([j["test_at_oof_thr"]["pr_auc"] for j in fin[f]]),
                                                                       np.std([j["test_at_oof_thr"]["pr_auc"] for j in fin[f]])))]:
        v = [get(f) for f in FAMS]
        ax.bar(x + off, [a for a, _ in v], .38, yerr=[b for _, b in v], color=col, label=lab, capsize=3)
    ax.set_xticks(x, [NAME[f] for f in FAMS])
    ax.set_ylabel("PR-AUC test")
    ax.set_title("Étude 1 contre étude 2 (l'étude 2 entraîne sur 21 patients au lieu de 16)")
    ax.legend()
    ax.grid(axis="y", alpha=.3)
    save(fig, "E2_04_study1_vs_study2")


def fig_bbb():
    bbb = bp.bbb_records()[0]
    rows = {}
    for f in FAMS:
        acc = {"b": [0, 0], "o": [0, 0]}
        for p in sorted((E2 / "final").glob(f"{f}__seed*_errors_by_record.csv")):
            for r in csv.DictReader(p.open(encoding="utf-8")):
                g = "b" if r["record"] in bbb else "o"
                acc[g][0] += int(r["fn"])
                acc[g][1] += int(r["n_abnormal"])
        rows[f] = [1 - acc[g][0] / acc[g][1] for g in ("b", "o")]
    x = np.arange(len(FAMS))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(x - .2, [rows[f][0] for f in FAMS], .38, color="#e4572e", label="4 patients à bloc de branche")
    ax.bar(x + .2, [rows[f][1] for f in FAMS], .38, color="#1f8a8a", label="autres patients")
    ax.set_xticks(x, [NAME[f] for f in FAMS])
    ax.set_ylabel("rappel au seuil OOF")
    ax.set_title("Étude 2 : rappel selon le type de patient")
    ax.legend()
    ax.grid(axis="y", alpha=.3)
    save(fig, "E2_05_recall_bbb")


def main():
    fin = finals()
    fig_cv_folds()
    fig_test_seeds(fin)
    fig_pr_curves(fin)
    fig_study1_vs_2(fin)
    fig_bbb()
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
