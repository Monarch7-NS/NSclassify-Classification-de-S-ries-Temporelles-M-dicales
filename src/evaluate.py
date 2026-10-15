"""Métriques (anormal = positif), seuil choisi sur validation, robustesse au bruit, analyse d'erreurs, coût, figures."""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)


def compute_metrics(y, p, thr):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fn = int((~pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    tn = int((~pred & (y == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "threshold": float(thr),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": 2 * prec * sens / max(prec + sens, 1e-12),
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": 0.5 * (sens + spec),
        "mcc": float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0,
        "brier": float(np.mean((p - y) ** 2)),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
    }


def select_threshold(y, p):
    """Seuil maximisant le F1 sur la validation."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[np.argmax(f1)])


def noise_robustness(predict, X, RR, y, thr, sigmas=(0.02, 0.05), seed=42):
    """Bruit gaussien ajouté aux battements de test, modèle et seuil inchangés."""
    rng = np.random.default_rng(seed)
    out = {"clean": compute_metrics(y, predict(X, RR), thr)}
    for s in sigmas:
        Xn = X + rng.normal(0.0, s, X.shape).astype(np.float32)
        out[f"sigma_{s}"] = compute_metrics(y, predict(Xn, RR), thr)
    return out


def plot_confusion(m, path, title):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.1%})", ha="center", va="center")
    ax.set_xticks([0, 1], ["pred normal", "pred anormal"])
    ax.set_yticks([0, 1], ["normal", "anormal"])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pr(curves, path):
    """curves: {nom: (y, p)}"""
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, (y, p) in curves.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{name} (AP={average_precision_score(y, p):.3f})")
    ax.set_xlabel("Rappel (sensibilité)")
    ax.set_ylabel("Précision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def threshold_study(y, p, target=0.90):
    """Trois seuils, tous déterminés sur le split fourni (la validation) : 0.5, max-F1, sensibilité >= cible."""
    fpr, tpr, thr = roc_curve(y, p)
    i = int(np.argmax(tpr >= target))
    return {"default": 0.5, "max_f1": select_threshold(y, p), "sens_target": float(thr[i])}


def plot_roc_pr(y, p, thresholds, path, title=""):
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    axes[0].set(xlabel="1 - spécificité", ylabel="sensibilité", title=f"ROC (AUC={roc_auc_score(y, p):.3f})")
    axes[1].plot(rec, prec)
    axes[1].set(xlabel="rappel", ylabel="précision", title=f"PR (AP={average_precision_score(y, p):.3f})")
    for name, t in thresholds.items():
        pred = p >= t
        axes[0].plot((pred & (y == 0)).sum() / max((y == 0).sum(), 1), (pred & (y == 1)).sum() / max((y == 1).sum(), 1), "o", label=name)
        axes[1].plot((pred & (y == 1)).sum() / max((y == 1).sum(), 1), (pred & (y == 1)).sum() / max(pred.sum(), 1), "o", label=name)
    axes[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_threshold_curves(y, p, path, title=""):
    ts = np.linspace(0.02, 0.98, 97)
    m = [compute_metrics(y, p, t) for t in ts]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for k in ("sensitivity", "specificity", "precision", "f1"):
        ax.plot(ts, [x[k] for x in m], label=k)
    ax.set(xlabel="seuil", title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def error_tables(record_id, symbol, y, pred):
    """Taux de FP/FN par enregistrement et taux d'erreur par type de battement (symbole MIT-BIH)."""
    import pandas as pd
    df = pd.DataFrame({"record": record_id, "symbol": symbol, "y": y, "pred": pred.astype(int)})
    df["fp"] = ((df.pred == 1) & (df.y == 0)).astype(int)
    df["fn"] = ((df.pred == 0) & (df.y == 1)).astype(int)
    by_rec = df.groupby("record").agg(n=("y", "size"), n_abnormal=("y", "sum"), fp=("fp", "sum"), fn=("fn", "sum"))
    by_rec["fp_rate"] = by_rec.fp / (by_rec.n - by_rec.n_abnormal).clip(lower=1)
    by_rec["fn_rate"] = by_rec.fn / by_rec.n_abnormal.clip(lower=1)
    df["err"] = (df.pred != df.y).astype(int)
    by_sym = df.groupby("symbol").agg(n=("y", "size"), errors=("err", "sum"))
    by_sym["error_rate"] = by_sym.errors / by_sym.n
    return by_rec.reset_index(), by_sym.reset_index()


def plot_error_examples(X, y, p, thr, path, k=4):
    """Les k faux positifs et k faux négatifs les plus confiants (fenêtres normalisées)."""
    pred = p >= thr
    fp = np.flatnonzero(pred & (y == 0))
    fn = np.flatnonzero(~pred & (y == 1))
    fp = fp[np.argsort(-p[fp])][:k]
    fn = fn[np.argsort(p[fn])][:k]
    fig, axes = plt.subplots(2, k, figsize=(3 * k, 5), squeeze=False)
    for row, (idx, name) in enumerate(((fp, "FP"), (fn, "FN"))):
        for col in range(k):
            ax = axes[row, col]
            if col < len(idx):
                ax.plot(X[idx[col]], lw=1)
                ax.set_title(f"{name} p={p[idx[col]]:.2f}", fontsize=9)
            ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def time_callable(fn, repeats=100, warmup=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))
