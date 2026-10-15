import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.inspection import permutation_importance

from .legacy_data import FEATURE_NAMES


def plot_logistic_coefficients(pipe, path, names=None):
    names = names or FEATURE_NAMES
    coef = pipe.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(np.array(names)[order], coef[order],
            color=["#e4572e" if c > 0 else "#1f8a8a" for c in coef[order]])
    ax.set_xlabel("Coefficient standardisé (log-odds d'anormalité)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_permutation_importance(rf, F, y, path, n=5000, seed=42, names=None):
    names = names or FEATURE_NAMES
    idx = np.random.default_rng(seed).choice(len(y), min(n, len(y)), replace=False)
    r = permutation_importance(
        rf, F[idx], y[idx], scoring="average_precision",
        n_repeats=5, random_state=seed, n_jobs=-1,
    )
    order = np.argsort(r.importances_mean)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(np.array(names)[order], r.importances_mean[order],
            xerr=r.importances_std[order], color="#1f8a8a")
    ax.set_xlabel("Baisse de PR-AUC après permutation (validation)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cnn_saliency(model, X, device):
    model.eval()
    x = torch.tensor(X, device=device).unsqueeze(1).requires_grad_(True)
    model(x).sum().backward()
    return x.grad.abs().squeeze(1).cpu().numpy()


def plot_saliency_cases(model, device, X, y, p, thr, path):
    """Un exemple TP / TN / FP / FN (le plus confiant dans son cas) avec saillance de gradient."""
    pred = p >= thr
    cases = {
        "Vrai positif": (pred & (y == 1), np.argmax),
        "Vrai négatif": (~pred & (y == 0), np.argmin),
        "Faux positif": (pred & (y == 0), np.argmax),
        "Faux négatif": (~pred & (y == 1), np.argmin),
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    t = np.arange(X.shape[1])
    for ax, (name, (mask, pick)) in zip(axes.ravel(), cases.items()):
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            ax.set_title(f"{name} : aucun")
            continue
        i = idx[pick(p[idx])]
        sal = cnn_saliency(model, X[i:i + 1], device)[0]
        ax.plot(t, X[i], color="#232b3a", lw=1)
        ax.pcolormesh(
            np.append(t, t[-1] + 1), ax.get_ylim(), (sal / (sal.max() + 1e-12))[None, :],
            cmap="Reds", alpha=0.35, shading="flat",
        )
        ax.set_title(f"{name} (p={p[i]:.2f})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Modèles de l'étude (forward(x, rr)) : gradient x entrée / gradients intégrés, CAM
# ---------------------------------------------------------------------------

def input_attribution(model, X, RR, device, use_rr, ig_steps=0):
    """|gradient| (ig_steps=0) ou |gradients intégrés vs ligne de base nulle| par échantillon temporel. X: [n, L]."""
    model.eval()
    x = torch.tensor(X, device=device).unsqueeze(1)
    rr = torch.tensor(RR, device=device) if use_rr else None
    with torch.backends.cudnn.flags(enabled=False):
        if ig_steps:
            total = torch.zeros_like(x)
            for a in torch.linspace(1.0 / ig_steps, 1.0, ig_steps):
                xi = (a * x).clone().requires_grad_(True)
                model(xi, rr).sum().backward()
                total += xi.grad
            attr = x * total / ig_steps
        else:
            xi = x.clone().requires_grad_(True)
            model(xi, rr).sum().backward()
            attr = xi.grad
    model.zero_grad(set_to_none=True)
    return attr.abs().squeeze(1).cpu().numpy()


def plot_attribution_cases(model, device, X, RR, y, p, thr, path, use_rr, ig_steps=20, title=""):
    """TP / TN / FP / FN (le plus confiant dans son cas) avec attribution par gradients intégrés."""
    pred = p >= thr
    cases = {
        "Vrai positif": (pred & (y == 1), np.argmax),
        "Vrai négatif": (~pred & (y == 0), np.argmin),
        "Faux positif": (pred & (y == 0), np.argmax),
        "Faux négatif": (~pred & (y == 1), np.argmin),
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    t = np.arange(X.shape[1])
    for ax, (name, (mask, pick)) in zip(axes.ravel(), cases.items()):
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            ax.set_title(f"{name} : aucun")
            continue
        i = idx[pick(p[idx])]
        a = input_attribution(model, X[i:i + 1], RR[i:i + 1], device, use_rr, ig_steps)[0]
        ax.plot(t, X[i], color="#232b3a", lw=1)
        ax.pcolormesh(np.append(t, t[-1] + 1), ax.get_ylim(), (a / (a.max() + 1e-12))[None, :],
                      cmap="Reds", alpha=0.35, shading="flat")
        ax.set_title(f"{name} (p={p[i]:.2f})")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
