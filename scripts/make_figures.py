"""Génère toutes les figures de synthèse à partir de results/experiments (index.json, history.csv, metrics.json, *_probs.npy).
Aucune valeur n'est saisie à la main. Les figures de test n'apparaissent que si le stage `final` existe.

Usage (depuis PROJET/) :  python -m scripts.make_figures
"""
import json
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_processing import SPLIT_NAMES, cache_path, load_config  # noqa: E402
from src.evaluate import compute_metrics  # noqa: E402
from src.experiment_tracker import EXP_DIR, load_index  # noqa: E402

OUT = EXP_DIR.parent / "figures"
COLORS = {"mlp": "#1f77b4", "cnn": "#d62728", "cnn_lstm": "#2ca02c", "lstm": "#8c564b", "lr": "#9467bd", "rf": "#ff7f0e"}
FAMILIES = ["mlp", "cnn", "cnn_lstm", "lstm", "lr", "rf"]
FS = 360.0
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=140)
    plt.close(fig)
    print("  ok", name)


def load_split(cfg, split):
    with np.load(cache_path(cfg, split)) as d:
        return {k: d[k] for k in d.files}


def run_dir(rid):
    return EXP_DIR / rid


def metrics_of(rid):
    return json.loads((run_dir(rid) / "metrics.json").read_text(encoding="utf-8"))


def val_threshold(rid):
    m = metrics_of(rid)
    return m["val_metrics"][m.get("primary_threshold", "max_f1")]["threshold"]


# ---------- données ----------

def fig_data_overview(cfg, data):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1, 3.2, 1.6]})
    ax = axes[0]
    rows = [(s, int((data[s]["y"] == 0).sum()), int((data[s]["y"] == 1).sum())) for s in SPLIT_NAMES]
    x = np.arange(len(rows))
    ax.bar(x, [r[1] for r in rows], label="normal", color="#4c78a8")
    ax.bar(x, [r[2] for r in rows], bottom=[r[1] for r in rows], label="anormal", color="#e45756")
    for i, r in enumerate(rows):
        ax.text(i, r[1] + r[2], f"{r[2] / (r[1] + r[2]):.1%} anormal", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_title("Battements par split")
    ax.legend()

    ax = axes[1]
    recs = []
    for s in SPLIT_NAMES:
        d = data[s]
        for r in np.unique(d["record_id"]):
            m = d["record_id"] == r
            recs.append((s, str(r), int((d["y"][m] == 0).sum()), int((d["y"][m] == 1).sum())))
    recs.sort(key=lambda t: (SPLIT_NAMES.index(t[0]), t[1]))
    pal = {"train": "#4c78a8", "val": "#f58518", "test": "#54a24b"}
    x = np.arange(len(recs))
    frac = [r[3] / max(r[2] + r[3], 1) for r in recs]
    ax.bar(x, frac, color=[pal[r[0]] for r in recs])
    ax.set_xticks(x, [r[1] for r in recs], rotation=90, fontsize=7)
    ax.set_ylabel("part de battements anormaux")
    ax.set_title("Prévalence d'anormaux par enregistrement (couleur = split)")
    ax.legend(handles=[Patch(color=c, label=s) for s, c in pal.items()])

    ax = axes[2]
    sym = pd.concat([pd.Series(data[s]["symbol"][data[s]["y"] == 1]) for s in SPLIT_NAMES]).value_counts()
    ax.barh(sym.index.astype(str), sym.values, color="#e45756")
    ax.set_xscale("log")
    ax.set_title("Types de battements anormaux (échelle log)")
    save(fig, "fig01_data_overview")


def fig_beats(cfg, data):
    from src.datasets import prepare
    p = prepare("P1", cfg["defaults"]["deep"]["window"], cfg)
    X, y = p.X["train"], p.y["train"]
    t = (np.arange(X.shape[1]) - p.centre) / FS * 1000
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for j, (lab, c) in enumerate([("normal", "#4c78a8"), ("anormal", "#e45756")]):
        Xc = X[y == j]
        ax = axes[0, j]
        ax.plot(t, Xc.mean(0), color=c, lw=2, label="moyenne")
        ax.fill_between(t, Xc.mean(0) - Xc.std(0), Xc.mean(0) + Xc.std(0), color=c, alpha=0.25, label="± écart-type")
        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.set_title(f"{lab} (train, P1) — n={len(Xc)}")
        ax.legend()
        ax = axes[1, j]
        for i in rng.choice(len(Xc), 8, replace=False):
            ax.plot(t, Xc[i], lw=0.9, alpha=0.8)
        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.set_title(f"{lab} : 8 battements tirés au hasard")
        ax.set_xlabel("temps par rapport à R (ms)")
    save(fig, "fig02_beats_normal_vs_abnormal")


# ---------- baseline ----------

def baseline_rows(df):
    b = df[df.stage == "baseline"].set_index("family")
    return [(f, b.loc[f, "id"]) for f in FAMILIES if f in b.index]


def fig_baseline_metrics(df, yv):
    b = df[df.stage == "baseline"].set_index("family").reindex(FAMILIES).dropna(subset=["id"])
    mets = [("val_pr_auc", "PR-AUC (critère)"), ("val_roc_auc", "ROC-AUC"), ("val_f1", "F1 @ max-F1"),
            ("val_mcc", "MCC"), ("val_balanced_accuracy", "Balanced acc."), ("val_sensitivity", "Sensibilité"),
            ("val_specificity", "Spécificité"), ("val_precision", "Précision")]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    for ax, (k, lab) in zip(axes.ravel(), mets):
        v = pd.to_numeric(b[k])
        ax.bar(v.index, v.values, color=[COLORS[f] for f in v.index])
        for i, val in enumerate(v.values):
            ax.text(i, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        if k == "val_pr_auc":
            ax.axhline(yv.mean(), color="k", ls="--", lw=1, label=f"hasard = prévalence {yv.mean():.3f}")
            ax.legend(fontsize=8)
        ax.set_title(lab)
        ax.set_ylim(min(0, v.min() - 0.05), 1.0)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Baseline — validation (1 seed) ; les différences < dispersion entre seeds ne sont pas significatives")
    save(fig, "fig03_baseline_metrics_val")


def fig_pr_roc(rows, y, name, title):
    """rows: [(famille, id, proba)]"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for f, rid, p in rows:
        prec, rec, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(rec, prec, color=COLORS[f], label=f"{f} AP={average_precision_score(y, p):.3f}")
        axes[1].plot(fpr, tpr, color=COLORS[f], label=f"{f} AUC={roc_auc_score(y, p):.3f}")
    axes[0].axhline(y.mean(), color="k", ls="--", lw=1, label=f"hasard ({y.mean():.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="rappel", ylabel="précision", title="Courbes précision-rappel")
    axes[1].set(xlabel="1 - spécificité", ylabel="sensibilité", title="Courbes ROC")
    for a in axes:
        a.legend()
    fig.suptitle(title)
    save(fig, name)


def fig_threshold_scan(rows, y, name, title):
    ts = np.linspace(0.01, 0.99, 99)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for f, rid, p in rows:
        f1 = [compute_metrics(y, p, t)["f1"] for t in ts]
        axes[0].plot(ts, f1, color=COLORS[f], label=f)
        thr = val_threshold(rid)
        axes[0].plot(thr, compute_metrics(y, p, thr)["f1"], "o", color=COLORS[f], ms=8, mec="k")
        m = [compute_metrics(y, p, t) for t in ts]
        axes[1].plot([x["sensitivity"] for x in m], [x["specificity"] for x in m], color=COLORS[f], label=f)
        axes[1].plot(compute_metrics(y, p, thr)["sensitivity"], compute_metrics(y, p, thr)["specificity"], "o",
                     color=COLORS[f], ms=8, mec="k")
    axes[0].set(xlabel="seuil", ylabel="F1", title="Balayage du seuil (point = seuil max-F1 choisi sur la validation)")
    axes[1].set(xlabel="sensibilité", ylabel="spécificité", title="Compromis sensibilité / spécificité")
    for a in axes:
        a.legend()
    fig.suptitle(title)
    save(fig, name)


def fig_confusions(rows, y, name, title):
    fig, axes = plt.subplots(1, len(rows), figsize=(3.6 * len(rows), 3.9))
    axes = np.atleast_1d(axes)
    for ax, (f, rid, p) in zip(axes, rows):
        m = compute_metrics(y, p, val_threshold(rid))
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.1%})", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1], ["pred N", "pred A"])
        ax.set_yticks([0, 1], ["vrai N", "vrai A"])
        ax.grid(False)
        ax.set_title(f"{f}  thr={m['threshold']:.3f}")
    fig.suptitle(title)
    save(fig, name)


def fig_score_hist(rows, y, name, title):
    fig, axes = plt.subplots(1, len(rows), figsize=(3.6 * len(rows), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    bins = np.linspace(0, 1, 41)
    for ax, (f, rid, p) in zip(axes, rows):
        ax.hist(p[y == 0], bins=bins, alpha=0.6, color="#4c78a8", label="normal", density=True)
        ax.hist(p[y == 1], bins=bins, alpha=0.6, color="#e45756", label="anormal", density=True)
        ax.axvline(val_threshold(rid), color="k", ls="--", lw=1, label="seuil")
        ax.set_title(f)
        ax.set_yscale("log")
        ax.set_xlabel("probabilité prédite")
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    save(fig, name)


def fig_calibration(rows, y, name, title):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    edges = np.linspace(0, 1, 11)
    for f, rid, p in rows:
        idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 20:
                xs.append(p[m].mean())
                ys.append(y[m].mean())
        ax.plot(xs, ys, "o-", color=COLORS[f], label=f)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="probabilité moyenne prédite", ylabel="fréquence observée d'anormaux", title=title)
    ax.legend()
    save(fig, name)


def fig_errors_per_record(rows, d, name, title):
    recs = sorted(np.unique(d["record_id"]).tolist())
    y = d["y"]
    fp = np.full((len(rows), len(recs)), np.nan)
    fn = np.full_like(fp, np.nan)
    for i, (f, rid, p) in enumerate(rows):
        pred = p >= val_threshold(rid)
        for j, r in enumerate(recs):
            m = d["record_id"] == r
            neg, pos = (y[m] == 0).sum(), (y[m] == 1).sum()
            if neg:
                fp[i, j] = (pred[m] & (y[m] == 0)).sum() / neg
            if pos:
                fn[i, j] = (~pred[m] & (y[m] == 1)).sum() / pos
    fig, axes = plt.subplots(2, 1, figsize=(max(7, 0.6 * len(recs) + 3), 7.5))
    for ax, M, lab in ((axes[0], fp, "taux de faux positifs (FP / normaux)"), (axes[1], fn, "taux de faux négatifs (FN / anormaux)")):
        im = ax.imshow(M, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(recs)), recs, rotation=90)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.grid(False)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if M[i, j] > 0.5 else "black")
        ax.set_title(lab)
        fig.colorbar(im, ax=ax, fraction=0.02)
    fig.suptitle(title)
    save(fig, name)


def fig_learning_curves(df):
    deep = [(f, rid) for f, rid in baseline_rows(df) if f in ("mlp", "cnn", "cnn_lstm", "lstm")]
    fig, axes = plt.subplots(2, len(deep), figsize=(5.2 * len(deep), 7.5), squeeze=False)
    for j, (f, rid) in enumerate(deep):
        h = pd.read_csv(run_dir(rid) / "history.csv")
        best = int(df.loc[df.id == rid, "best_epoch"].iloc[0])
        ax = axes[0, j]
        ax.plot(h.epoch, h.train_loss, label="loss train", color="#4c78a8")
        ax.plot(h.epoch, h.val_loss, label="loss val", color="#e45756")
        ax.axvline(best, color="k", ls="--", lw=1, label=f"meilleure époque = {best}")
        ax.set(title=f"{f} — loss", xlabel="époque")
        ax.legend()
        ax = axes[1, j]
        ax.plot(h.epoch, h.val_pr_auc, color="#54a24b", label="PR-AUC val")
        ax.plot(h.epoch, h.val_roc_auc, color="#f58518", label="ROC-AUC val")
        ax.axvline(best, color="k", ls="--", lw=1)
        ax.set(title=f"{f} — AUC validation", xlabel="époque")
        ax.legend()
    fig.suptitle("Courbes d'apprentissage (baseline). La meilleure époque est choisie sur la PR-AUC de validation")
    save(fig, "fig10_learning_curves_baseline")


# ---------- facteurs ----------

def fig_factor_effects(df):
    g = df[df.stage.isin(["baseline", "factors"])]
    fams = [f for f in ("mlp", "cnn", "cnn_lstm", "lr", "rf") if (g[g.family == f].stage == "factors").any()]
    if not fams:
        return
    fig, axes = plt.subplots(1, len(fams), figsize=(6 * len(fams), 9), squeeze=False)
    for ax, f in zip(axes[0], fams):
        sub = g[g.family == f]
        ref = sub[sub.stage == "baseline"]
        if not len(ref):
            continue
        r0 = float(ref.val_pr_auc.iloc[0])
        s = sub[sub.stage == "factors"].assign(delta=lambda x: x.val_pr_auc - r0).sort_values("delta")
        ax.barh(s.factor, s.delta, color=np.where(s.delta >= 0, "#54a24b", "#e45756"))
        ax.axvline(0, color="k", lw=1)
        ax.set_title(f"{f} — écart de PR-AUC val vs baseline ({r0:.3f})")
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Effet de chaque facteur, un par un (1 seed : un écart petit n'est pas significatif)")
    save(fig, "fig11_factor_effects")


def fig_window_ablation(df):
    s = df[df.factor.isin(["baseline", "window=216", "window=280"]) & df.stage.isin(["baseline", "factors"])]
    s = s.assign(win=pd.to_numeric(s.window)).dropna(subset=["val_pr_auc"])
    if s.win.nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for f, g in s.groupby("family"):
        g = g.sort_values("win")
        ax.plot(g.win, g.val_pr_auc, "o-", color=COLORS.get(f), label=f)
    ax.set(xlabel="longueur de fenêtre (échantillons)", ylabel="PR-AUC validation", title="Ablation de la fenêtre (1 seed)")
    ax.set_xticks([216, 280, 360])
    ax.legend()
    save(fig, "fig12_window_ablation")


def fig_all_runs_strip(df):
    s = df[df.status == "ok"].dropna(subset=["val_pr_auc"])
    fig, ax = plt.subplots(figsize=(9, 5))
    rng = np.random.default_rng(0)
    for i, f in enumerate(FAMILIES):
        v = s[s.family == f].val_pr_auc.values
        if len(v):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, color=COLORS[f], alpha=0.7)
            ax.hlines(np.median(v), i - 0.3, i + 0.3, color="k")
    ax.set_xticks(range(len(FAMILIES)), FAMILIES)
    ax.set(ylabel="PR-AUC validation", title="Dispersion de toutes les configurations testées (trait = médiane)")
    save(fig, "fig13_all_runs_val_pr_auc")


def fig_cost(df):
    s = df[(df.stage == "baseline") & (df.status == "ok")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for _, r in s.iterrows():
        c = COLORS[r.family]
        axes[0].scatter(r.n_params, r.val_pr_auc, s=140, color=c, label=r.family)
        axes[1].scatter(r.epoch_time_s, r.val_pr_auc, s=140, color=c)
        axes[2].scatter(r.latency_ms_single, r.val_pr_auc, s=140, color=c)
    axes[0].set(xscale="log", xlabel="nombre de paramètres", ylabel="PR-AUC val", title="Performance vs taille")
    axes[1].set(xscale="log", xlabel="temps par époque (s)", title="Performance vs temps d'entraînement")
    axes[2].set(xscale="log", xlabel="latence par battement (ms)", title="Performance vs inférence")
    axes[0].legend()
    save(fig, "fig14_cost_vs_performance")


# ---------- final (test) ----------

def fig_final(df, cfg):
    fin = df[(df.stage == "final") & (df.status == "ok")]
    if not len(fin):
        print("  (stage final absent : figures de test non générées)")
        return
    fin = fin.assign(pr=pd.to_numeric(fin.test_pr_auc))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (k, lab) in zip(axes, [("test_pr_auc", "PR-AUC test"), ("test_f1", "F1 test"), ("test_mcc", "MCC test")]):
        for i, f in enumerate(FAMILIES):
            v = pd.to_numeric(fin[fin.family == f][k]).values
            if len(v):
                ax.bar(i, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0, color=COLORS[f], alpha=0.6, capsize=5)
                ax.scatter([i] * len(v), v, color="k", zorder=3, s=18)
        ax.set_xticks(range(len(FAMILIES)), FAMILIES)
        ax.set_title(f"{lab} (moyenne ± écart-type, points = seeds)")
    save(fig, "fig15_final_test_seeds")

    test = load_split(cfg, "test")
    y = test["y"]
    rows = []
    for f in FAMILIES:
        g = fin[fin.family == f]
        if len(g) and (run_dir(g.id.iloc[0]) / "test_probs.npy").exists():
            rows.append((f, g.id.iloc[0], np.load(run_dir(g.id.iloc[0]) / "test_probs.npy")))
    if rows:
        fig_pr_roc(rows, y, "fig16_final_test_pr_roc", "Test — 1ère seed de chaque famille")
        fig_errors_per_record(rows, test, "fig17_final_test_errors_per_record", "Test — erreurs par patient (seuil figé sur la validation)")
        fig_confusions(rows, y, "fig18_final_test_confusions", "Test — matrices de confusion (seuil figé sur la validation)")


def main():
    idx = load_index()
    if not idx:
        print("index vide")
        return
    cfg = load_config()
    df = pd.DataFrame(idx.values())
    for c in ("val_pr_auc", "val_roc_auc", "val_f1", "val_mcc", "val_balanced_accuracy", "val_sensitivity", "val_specificity",
              "val_precision", "n_params", "epoch_time_s", "latency_ms_single", "best_epoch"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    data = {s: load_split(cfg, s) for s in SPLIT_NAMES}
    yv = data["val"]["y"]
    base = baseline_rows(df)
    vrows = [(f, rid, np.load(run_dir(rid) / "val_probs.npy")) for f, rid in base if (run_dir(rid) / "val_probs.npy").exists()]

    steps = [
        ("fig01", lambda: fig_data_overview(cfg, data)),
        ("fig02", lambda: fig_beats(cfg, data)),
        ("fig03", lambda: fig_baseline_metrics(df, yv)),
        ("fig04", lambda: fig_pr_roc(vrows, yv, "fig04_baseline_pr_roc_val", "Baseline — validation")),
        ("fig05", lambda: fig_threshold_scan(vrows, yv, "fig05_baseline_threshold_scan_val", "Baseline — validation")),
        ("fig06", lambda: fig_confusions(vrows, yv, "fig06_baseline_confusions_val", "Baseline — validation (seuil max-F1)")),
        ("fig07", lambda: fig_score_hist(vrows, yv, "fig07_baseline_score_hist_val", "Distribution des scores par classe (validation)")),
        ("fig08", lambda: fig_calibration(vrows, yv, "fig08_baseline_calibration_val", "Calibration (validation)")),
        ("fig09", lambda: fig_errors_per_record(vrows, data["val"], "fig09_baseline_errors_per_record_val",
                                                "Validation — erreurs par patient (124, 203, 205, 207, 215)")),
        ("fig10", lambda: fig_learning_curves(df)),
        ("fig11", lambda: fig_factor_effects(df)),
        ("fig12", lambda: fig_window_ablation(df)),
        ("fig13", lambda: fig_all_runs_strip(df)),
        ("fig14", lambda: fig_cost(df)),
        ("fig15+", lambda: fig_final(df, cfg)),
    ]
    for name, fn in steps:
        try:
            fn()
        except Exception:
            print("  ECHEC", name)
            traceback.print_exc()
    print("figures écrites dans", OUT)


if __name__ == "__main__":
    main()
