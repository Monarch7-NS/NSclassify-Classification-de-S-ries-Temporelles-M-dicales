"""Entraînement et évaluation d'une expérience décrite par un dictionnaire de configuration (voir configs/experiment.json).

Règles appliquées ici :
  - la sélection (early stopping, scheduler, seuils) ne regarde que la validation ;
  - le test n'est évalué que si cfg["stage"] == "final", une seule fois, avec des seuils figés sur la validation ;
  - l'augmentation n'existe que sur le split train (ECGData.augment_batch lève une erreur sinon).
"""
import itertools
import json
import time
import traceback

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import evaluate as ev
from . import experiment_tracker as tr
from .data_processing import SPLITS_PATH, load_config
from .datasets import make_data, prepare
from .features import FEATURE_NAMES
from .interpretability import plot_attribution_cases, plot_logistic_coefficients, plot_permutation_importance
from .losses import build_loss
from .models import build_model, count_params

N_RR = 5


# ---------------------------------------------------------------------------
# Deep learning
# ---------------------------------------------------------------------------

def build_optimizer(params, spec):
    name, lr, wd = spec["name"], spec["lr"], spec.get("weight_decay", 0.0)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)       # L2 couplé au gradient adaptatif
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)      # décroissance découplée
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=spec.get("momentum", 0.9), weight_decay=wd, nesterov=True)
    raise ValueError(name)


@torch.no_grad()
def predict_logits(model, X, RR, use_rr, device, bs=2048):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).unsqueeze(1).to(device)
        rb = torch.from_numpy(np.ascontiguousarray(RR[i:i + bs])).to(device) if use_rr else None
        out.append(model(xb, rb).float().cpu())
    return torch.cat(out).numpy()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_deep(cfg, prep, device, run_dir):
    seed = cfg["seed"]
    tr.set_seed(seed)
    use_rr = cfg.get("use_rr", False)
    m = cfg["model"]
    model = build_model(m["arch"], m.get("params"), length=prep.X["train"].shape[1],
                        n_rr=prep.RR["train"].shape[1] if use_rr else 0).to(device)
    loss_fn = build_loss(cfg["loss"], prep.y["train"]).to(device)
    opt = build_optimizer(model.parameters(), cfg["optimizer"])
    epochs = cfg["epochs"]
    sched = None
    if cfg.get("scheduler") == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    elif cfg.get("scheduler") == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    data = make_data(prep)
    loader = data["train"].loader(cfg["batch_size"], shuffle=True, weighted=cfg.get("sampler") == "weighted", seed=seed)
    clip = cfg.get("clip_grad", 0.0)
    y_val = prep.y["val"]
    yv_t = torch.from_numpy(y_val.astype(np.float32)).to(device)

    history, best, best_state, best_epoch, bad, status = [], -np.inf, None, 0, 0, "ok"
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        tot, n, gn_sum, nb = 0.0, 0, 0.0, 0
        for xb, rb, yb in loader:
            xb, rb, yb = xb.to(device), rb.to(device), yb.to(device)
            if cfg.get("augment"):
                xb = data["train"].augment_batch(xb, cfg.get("augment_cfg"))
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb, rb if use_rr else None), yb)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip > 0 else float("inf"))
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
            gn_sum += float(gn)
            nb += 1
        train_loss = tot / n
        if not np.isfinite(train_loss):
            status = "diverged"
            break
        logits = predict_logits(model, prep.X["val"], prep.RR["val"], use_rr, device)
        lt = torch.from_numpy(logits).to(device)
        val_loss = float(loss_fn(lt, yv_t))
        val_bce = float(torch.nn.functional.binary_cross_entropy_with_logits(lt, yv_t))
        m05 = ev.compute_metrics(y_val, sigmoid(logits), 0.5)
        pr_auc = m05["pr_auc"]
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_bce": val_bce,
            "val_pr_auc": pr_auc, "val_roc_auc": m05["roc_auc"], "val_f1_at_0.5": m05["f1"],
            "val_sens_at_0.5": m05["sensitivity"], "val_spec_at_0.5": m05["specificity"],
            "lr": opt.param_groups[0]["lr"], "grad_norm": gn_sum / max(nb, 1),
            "epoch_time_s": time.perf_counter() - t0,
        })
        if sched is not None:
            sched.step(pr_auc) if cfg["scheduler"] == "plateau" else sched.step()
        if pr_auc > best:
            best, bad, best_epoch = pr_auc, 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    train_time = time.perf_counter() - t_start
    if best_state is None:
        raise RuntimeError(f"aucun modèle valide ({status})")
    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "best.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    plot_history(history, best_epoch, run_dir / "learning_curves.png", run_dir.name)

    def predict_arrays(X, RR):
        return sigmoid(predict_logits(model, X, RR, use_rr, device))

    def predict(split):
        return predict_arrays(prep.X[split], prep.RR[split])

    extra = {
        "best_epoch": best_epoch, "epochs_run": len(history), "train_time_s": train_time,
        "epoch_time_s": float(np.mean([h["epoch_time_s"] for h in history])),
        "n_params": count_params(model), "status": status,
        "augmented_batches_train": data["train"].n_augmented_batches,
        "augmented_batches_val_test": data["val"].n_augmented_batches + data["test"].n_augmented_batches,
        "overfit_gap_last_epoch": history[-1]["val_loss"] - history[-1]["train_loss"],
    }

    # coût d'inférence (CPU)
    model.eval().cpu()
    x1 = torch.from_numpy(prep.X["val"][:1]).unsqueeze(1)
    r1 = torch.from_numpy(prep.RR["val"][:1]) if use_rr else None
    xb = torch.from_numpy(prep.X["val"][:1024]).unsqueeze(1)
    rb = torch.from_numpy(prep.RR["val"][:1024]) if use_rr else None
    with torch.no_grad():
        extra["latency_ms_single"] = 1000 * ev.time_callable(lambda: model(x1, r1), repeats=100)
        extra["throughput_beats_s"] = len(xb) / ev.time_callable(lambda: model(xb, rb), repeats=5, warmup=1)
    extra["model_bytes"] = (run_dir / "best.pt").stat().st_size
    model.to(device)
    return {"model": model, "predict": predict, "predict_arrays": predict_arrays, "extra": extra, "kind": "dl",
            "use_rr": use_rr}


def plot_history(history, best_epoch, path, title):
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(h.epoch, h.train_loss, label="train (objectif)")
    axes[0].plot(h.epoch, h.val_loss, label="val (objectif)")
    axes[0].plot(h.epoch, h.val_bce, "--", label="val BCE")
    axes[0].axvline(best_epoch, color="gray", ls=":")
    axes[0].set(xlabel="époque", ylabel="perte")
    axes[0].legend(fontsize=7)
    axes[1].plot(h.epoch, h.val_pr_auc, label="val PR-AUC")
    axes[1].plot(h.epoch, h.val_roc_auc, label="val ROC-AUC")
    axes[1].axvline(best_epoch, color="gray", ls=":")
    axes[1].set(xlabel="époque")
    axes[1].legend(fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# sklearn (LR, RF) : mini-grille évaluée sur la validation
# ---------------------------------------------------------------------------

def _grid(spec):
    keys = list(spec)
    for combo in itertools.product(*[spec[k] for k in keys]):
        yield dict(zip(keys, combo))


def fit_sklearn(cfg, prep, run_dir):
    fam = cfg["model"]["family"]
    cols = slice(0, None) if cfg.get("use_rr", True) else slice(N_RR, None)
    Ftr, Fva = prep.F["train"][:, cols], prep.F["val"][:, cols]
    seed = cfg["seed"]

    def make(params):
        if fam == "lr":
            return Pipeline([("scale", StandardScaler()),
                             ("clf", LogisticRegression(solver="liblinear", max_iter=3000, **params))])
        return RandomForestClassifier(max_features="sqrt", n_jobs=-1, random_state=seed, **params)

    log, best, best_params, best_model = [], -np.inf, None, None
    t0 = time.perf_counter()
    for params in _grid(cfg["model"]["grid"]):
        mdl = make(params).fit(Ftr, prep.y["train"])
        ap = average_precision_score(prep.y["val"], mdl.predict_proba(Fva)[:, 1])
        log.append({**params, "val_pr_auc": ap})
        if ap > best:
            best, best_params, best_model = ap, params, mdl
    train_time = time.perf_counter() - t0
    pd.DataFrame(log).to_csv(run_dir / "search_log.csv", index=False)
    joblib.dump(best_model, run_dir / "model.joblib")

    def predict(split):
        return best_model.predict_proba(prep.F[split][:, cols])[:, 1]

    extra = {"train_time_s": train_time, "best_params": json.dumps(best_params), "status": "ok",
             "n_grid_points": len(log), "model_bytes": (run_dir / "model.joblib").stat().st_size,
             "latency_ms_single": 1000 * ev.time_callable(lambda: best_model.predict_proba(Fva[:1]), repeats=50),
             "throughput_beats_s": 5000 / ev.time_callable(lambda: best_model.predict_proba(Fva[:5000]), repeats=3, warmup=1)}
    if fam == "lr":
        extra["n_params"] = int(best_model.named_steps["clf"].coef_.size + 1)
    else:
        extra["n_params"] = int(sum(t.tree_.node_count for t in best_model.estimators_))
    return {"model": best_model, "predict": predict, "extra": extra, "kind": fam, "cols": cols}


# ---------------------------------------------------------------------------
# Évaluation commune
# ---------------------------------------------------------------------------

def evaluate_run(cfg, prep, fit, run_dir, device, final):
    sel = load_config()["selection"]
    primary = sel["threshold_rule"]
    y_val = prep.y["val"]
    val_p = fit["predict"]("val")
    thresholds = ev.threshold_study(y_val, val_p, sel["sensitivity_target"])
    result = {
        "primary_threshold": primary,
        "thresholds_from_val": thresholds,
        "val_metrics": {k: ev.compute_metrics(y_val, val_p, t) for k, t in thresholds.items()},
    }
    np.save(run_dir / "val_probs.npy", val_p.astype(np.float32))
    thr_p = thresholds[primary]
    ev.plot_confusion(result["val_metrics"][primary], run_dir / "val_confusion.png", f"val @ {primary}")
    ev.plot_roc_pr(y_val, val_p, thresholds, run_dir / "val_roc_pr.png", run_dir.name)
    ev.plot_threshold_curves(y_val, val_p, run_dir / "val_threshold_curves.png", "validation")

    if final:
        y_te = prep.y["test"]
        te_p = fit["predict"]("test")
        result["test_metrics"] = {k: ev.compute_metrics(y_te, te_p, t) for k, t in thresholds.items()}
        np.save(run_dir / "test_probs.npy", te_p.astype(np.float32))
        pred = te_p >= thr_p
        by_rec, by_sym = ev.error_tables(prep.meta["test"]["record_id"], prep.meta["test"]["symbol"], y_te, pred)
        by_rec.to_csv(run_dir / "test_errors_by_record.csv", index=False)
        by_sym.to_csv(run_dir / "test_errors_by_symbol.csv", index=False)
        ev.plot_confusion(result["test_metrics"][primary], run_dir / "test_confusion.png", f"test @ {primary}")
        ev.plot_roc_pr(y_te, te_p, thresholds, run_dir / "test_roc_pr.png", run_dir.name)
        ev.plot_error_examples(prep.X["test"], y_te, te_p, thr_p, run_dir / "test_error_examples.png")
        final_analysis(cfg, prep, fit, te_p, thr_p, run_dir, device, result)
    return result


def final_analysis(cfg, prep, fit, te_p, thr, run_dir, device, result):
    kind = fit["kind"]
    names = FEATURE_NAMES if cfg.get("use_rr", True) else FEATURE_NAMES[N_RR:]
    if kind == "lr":
        plot_logistic_coefficients(fit["model"], run_dir / "interp_coefficients.png", names=names)
    elif kind == "rf":
        plot_permutation_importance(fit["model"], prep.F["val"][:, fit["cols"]], prep.y["val"],
                                    run_dir / "interp_permutation.png", names=names)
    else:
        rng = np.random.default_rng(cfg["seed"])
        sub = rng.choice(len(te_p), 4000, replace=False)
        plot_attribution_cases(fit["model"], device, prep.X["test"][sub], prep.RR["test"][sub], prep.y["test"][sub],
                               te_p[sub], thr, run_dir / "interp_attribution.png", fit["use_rr"], title=run_dir.name)
        rob = {}
        for s in (0.0, 0.05, 0.1, 0.2):
            Xn = prep.X["test"] + rng.normal(0, s, prep.X["test"].shape).astype(np.float32) if s else prep.X["test"]
            rob[f"sigma_{s}"] = ev.compute_metrics(prep.y["test"], fit["predict_arrays"](Xn, prep.RR["test"]), thr)
        result["noise_robustness_test"] = rob


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def run_experiment(cfg, force=False, device=None):
    exp_id = cfg.get("id") or tr.config_id(cfg)
    run_dir = tr.EXP_DIR / exp_id
    final = cfg.get("stage") == "final"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        prev = json.loads(metrics_path.read_text(encoding="utf-8"))
        if final and "test_metrics" in prev:
            if force:
                raise PermissionError("le test a déjà été évalué pour cette configuration ; ré-évaluation refusée")
            return prev
        if not force and prev.get("status") == "ok":
            return prev
    run_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    try:
        prep = prepare(cfg["preprocess"], cfg["window"])
        fam = cfg["model"]["family"]
        fit = fit_sklearn(cfg, prep, run_dir) if fam in ("lr", "rf") else fit_deep(cfg, prep, device, run_dir)
        result = evaluate_run(cfg, prep, fit, run_dir, device, final)
        result.update(fit["extra"])
        (run_dir / "run_info.json").write_text(
            json.dumps(tr.run_info(cfg, device, prep.info, tr.file_sha1(SPLITS_PATH)), indent=2, default=str),
            encoding="utf-8")
    except Exception:
        result = {"status": "failed", "error": traceback.format_exc()}
    result["id"] = exp_id
    metrics_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    tr.update_index(exp_id, tr.flat_row(exp_id, cfg, result))
    return result
