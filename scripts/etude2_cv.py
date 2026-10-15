"""Étude 2 (etude2_cv_groupee) : optimisation des hyperparamètres par validation croisée groupée par patient.

Différences avec l'étude 1 (results_rr/) :
  - sélection par CV groupée par patient sur train + validation (21 patients, 5 plis) au lieu d'une validation de 5 patients ;
  - grilles orientées régularisation (weight decay, dropout, taille) et stabilité (CNN-LSTM : lr plus bas, pooling) ;
  - tous les réseaux reçoivent les 3 RR (CNN-LSTM compris) ; prétraitement P1, fenêtre 360, perte bce_pw ;
  - seuil max-F1 déterminé sur les prédictions hors pli (OOF) ; 5 seeds en final ;
  - modèle final entraîné sur les 21 patients, nombre d'époques fixé (médiane des meilleures époques des plis).
Le test n'est lu qu'à l'étape `final`, une seule fois par (famille, seed). Ce n'est PAS une première lecture : voir le README.

Sous-commandes : search | select | final | report   (voir --help)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("EXPERIMENT_CONFIG", "configs/experiment_rr.json")

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import evaluate as ev  # noqa: E402
from src import experiment_tracker as tr  # noqa: E402
from src.augmentation import augment  # noqa: E402
from src.datasets import prepare  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.models import build_model, count_params  # noqa: E402

OUT = ROOT / "results_etude2"
RUNS, FINAL, TABLES = OUT / "runs", OUT / "final", OUT / "tables"
RR_COLS = [0, 1, 3]            # rr_prev, rr_next, rr_ratio_prev (colonnes de rr_matrix), comme l'étude 1
K = 5
SEARCH_SEED = 42
FINAL_SEEDS = [42, 43, 44, 45, 46]
FAMILIES = ("mlp", "cnn", "cnn_lstm", "lr", "rf")
DL = ("mlp", "cnn", "cnn_lstm")

BASE = {"epochs": 30, "patience": 5, "bs": 128, "lr": 1e-3, "wd": 1e-4, "clip": 0.0, "augment": False}
_CNN = {"kernel": 5, "bn": True, "act": "relu", "pool": "max", "dropout": 0.2, "head": "flatten"}
_LSTM = {"kernel": 5, "bn": True, "act": "relu", "cnn_dropout": 0.2, "hidden": 64, "layers": 1, "pooling": "last"}


def _c(name, family, arch, params, **kw):
    return {**BASE, "name": name, "family": family, "arch": arch, "params": params, **kw}


GRIDS = {
    "mlp": [
        _c("mlp_wd1e-4_do0.2", "mlp", "mlp_a", {"dropout": 0.2}),
        _c("mlp_wd1e-3_do0.2", "mlp", "mlp_a", {"dropout": 0.2}, wd=1e-3),
        _c("mlp_wd1e-2_do0.2", "mlp", "mlp_a", {"dropout": 0.2}, wd=1e-2),
        _c("mlp_wd1e-3_do0.4", "mlp", "mlp_a", {"dropout": 0.4}, wd=1e-3),
        _c("mlp_small_wd1e-3_do0.5", "mlp", "mlp_a", {"dropout": 0.5, "hidden": [64, 32]}, wd=1e-3),
        _c("mlp_small_wd1e-2_do0.4", "mlp", "mlp_a", {"dropout": 0.4, "hidden": [64, 32]}, wd=1e-2),
    ],
    "cnn": [
        _c("cnn_m_flat_wd1e-4_do0.2", "cnn", "cnn_m", dict(_CNN)),
        _c("cnn_m_flat_wd1e-3_do0.5", "cnn", "cnn_m", {**_CNN, "dropout": 0.5}, wd=1e-3),
        _c("cnn_s_flat_wd1e-3_do0.5", "cnn", "cnn_s", {**_CNN, "dropout": 0.5}, wd=1e-3),
        _c("cnn_m_gap_wd1e-3_do0.3", "cnn", "cnn_m", {**_CNN, "head": "gap", "dropout": 0.3}, wd=1e-3),
        _c("cnn_s_gap_wd1e-3_do0.3", "cnn", "cnn_s", {**_CNN, "head": "gap", "dropout": 0.3}, wd=1e-3),
        _c("cnn_m_flat_wd1e-2_do0.5", "cnn", "cnn_m", {**_CNN, "dropout": 0.5}, wd=1e-2),
        _c("cnn_m_flat_wd1e-3_do0.5_aug", "cnn", "cnn_m", {**_CNN, "dropout": 0.5}, wd=1e-3, augment=True),
    ],
    "cnn_lstm": [
        _c("lstm_lr1e-3_clip1", "cnn_lstm", "cnnlstm_s", dict(_LSTM), clip=1.0),
        _c("lstm_lr3e-4_clip1", "cnn_lstm", "cnnlstm_s", dict(_LSTM), clip=1.0, lr=3e-4),
        _c("lstm_lr3e-4_wd1e-3_hdo0.5", "cnn_lstm", "cnnlstm_s", {**_LSTM, "head_dropout": 0.5}, clip=1.0, lr=3e-4, wd=1e-3),
        _c("lstm_h32_lr3e-4_wd1e-3_hdo0.5", "cnn_lstm", "cnnlstm_s", {**_LSTM, "hidden": 32, "head_dropout": 0.5},
           clip=1.0, lr=3e-4, wd=1e-3),
        _c("lstm_meanpool_lr3e-4_clip0.5", "cnn_lstm", "cnnlstm_s", {**_LSTM, "pooling": "mean"}, clip=0.5, lr=3e-4),
        _c("lstm_meanpool_lr3e-4_wd1e-3_hdo0.5", "cnn_lstm", "cnnlstm_s", {**_LSTM, "pooling": "mean", "head_dropout": 0.5},
           clip=1.0, lr=3e-4, wd=1e-3),
    ],
    "lr": [{"name": f"lr_C{c}", "family": "lr", "C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "rf": [{"name": f"rf_d{d}_leaf{l}", "family": "rf", "max_depth": d, "min_samples_leaf": l}
           for d in (12, 20) for l in (1, 5)],
}


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------

class Data:
    """Pool (train + val = 21 patients) et test, P1 / fenêtre 360. RR brut (3 colonnes) + features LR/RF (14 colonnes)."""

    def __init__(self):
        p = prepare("P1", 360)
        cat = lambda d: np.concatenate([d["train"], d["val"]])  # noqa: E731
        self.X, self.F, self.y = cat(p.X), cat(p.F), cat(p.y)
        self.rec = np.concatenate([p.meta["train"]["record_id"], p.meta["val"]["record_id"]]).astype(str)
        self.sym = np.concatenate([p.meta["train"]["symbol"], p.meta["val"]["symbol"]]).astype(str)
        # p.RR : 3 colonnes [0,1,3] (experiment_rr.json), standardisées ; re-standardiser par pli est une transformation affine
        # identique à standardiser les valeurs brutes avec les stats du pli.
        self.RR = cat(p.RR)
        assert self.RR.shape[1] == len(RR_COLS), "EXPERIMENT_CONFIG doit être configs/experiment_rr.json"
        self.Xt, self.Ft, self.yt = p.X["test"], p.F["test"], p.y["test"]
        self.rect = p.meta["test"]["record_id"].astype(str)
        self.symt = p.meta["test"]["symbol"].astype(str)
        self.RRt = p.RR["test"]
        self.fold = self._folds()

    def _folds(self):
        """Affectation déterministe record -> pli, calculée à partir des seuls comptages d'annotations (pas du test)."""
        path = OUT / "folds.json"
        recs = sorted(set(self.rec.tolist()), key=int)
        if path.exists():
            m = json.loads(path.read_text(encoding="utf-8"))["record_to_fold"]
        else:
            info = {r: {"n": int((self.rec == r).sum()), "abn": int(self.y[self.rec == r].sum()),
                        "bbb": float(np.isin(self.sym[self.rec == r], ["L", "R"]).mean())} for r in recs}
            bbb = sorted([r for r in recs if info[r]["bbb"] > 0.5], key=lambda r: -info[r]["abn"])
            rest = sorted([r for r in recs if r not in bbb], key=lambda r: -info[r]["n"])
            load, m = [0] * K, {}
            for i, r in enumerate(bbb):                       # blocs de branche répartis un par pli
                m[r] = i % K
                load[i % K] += info[r]["n"]
            for r in rest:                                    # puis remplissage glouton par nombre de battements
                f = int(np.argmin(load))
                m[r] = f
                load[f] += info[r]["n"]
            OUT.mkdir(parents=True, exist_ok=True)
            summary = [{"fold": f, "records": [r for r in recs if m[r] == f],
                        "n": int(sum(info[r]["n"] for r in recs if m[r] == f)),
                        "abnormal": int(sum(info[r]["abn"] for r in recs if m[r] == f))} for f in range(K)]
            path.write_text(json.dumps({"record_to_fold": m, "bbb_records": bbb, "folds": summary, "info": info},
                                       indent=2), encoding="utf-8")
        return np.array([m[r] for r in self.rec])


def std_rr(train_rr, *others):
    mu, sd = train_rr.mean(0), train_rr.std(0) + 1e-8
    return [((a - mu) / sd).astype(np.float32) for a in (train_rr, *others)]


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, X, RR, bs=4096):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.from_numpy(X[i:i + bs]).unsqueeze(1), torch.from_numpy(RR[i:i + bs])).float())
    return torch.sigmoid(torch.cat(out)).numpy()


def train_dl(cfg, Xtr, RRtr, ytr, seed, epochs, es=None):
    """es = (X, RR, y) du pli d'arrêt précoce, ou None (nombre d'époques fixé). Retourne modèle, meilleure époque, statut."""
    tr.set_seed(seed)
    model = build_model(cfg["arch"], cfg["params"], length=Xtr.shape[1], n_rr=RRtr.shape[1])
    loss_fn = build_loss({"name": "bce_pw"}, ytr)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    Xt, Rt, yt = torch.from_numpy(Xtr).unsqueeze(1), torch.from_numpy(RRtr), torch.from_numpy(ytr.astype(np.float32))
    g = torch.Generator().manual_seed(seed)
    n, best, best_state, best_epoch, bad, status = len(yt), -np.inf, None, 0, 0, "ok"
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, cfg["bs"]):
            idx = perm[i:i + cfg["bs"]]
            xb = augment(Xt[idx]) if cfg["augment"] else Xt[idx]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb, Rt[idx]), yt[idx])
            loss.backward()
            if cfg["clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            tot += float(loss.detach()) * len(idx)
        if not np.isfinite(tot):
            status = "diverged"
            break
        if es is not None:
            ap = average_precision_score(es[2], predict(model, es[0], es[1]))
            if ap > best:
                best, bad, best_epoch = ap, 0, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
    if es is not None:
        if best_state is None:
            raise RuntimeError("aucun modèle valide")
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    return model, best_epoch, status


def make_sk(cfg, seed):
    if cfg["family"] == "lr":
        return Pipeline([("s", StandardScaler()), ("c", LogisticRegression(
            C=cfg["C"], class_weight="balanced", solver="liblinear", max_iter=3000))])
    return RandomForestClassifier(n_estimators=300, max_depth=cfg["max_depth"], min_samples_leaf=cfg["min_samples_leaf"],
                                  max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=seed)


# ---------------------------------------------------------------------------
# Sous-commandes
# ---------------------------------------------------------------------------

def ext(base, suffix):
    """Ajoute une extension sans toucher aux points du nom (Path.with_suffix écraserait '.2' de 'do0.2')."""
    return base.parent / (base.name + suffix)


def run_path(cfg, k):
    return RUNS / f"{cfg['family']}__{cfg['name']}__fold{k}__seed{SEARCH_SEED}"


def cmd_search(a, d):
    RUNS.mkdir(parents=True, exist_ok=True)
    cfgs = GRIDS[a.family]
    if a.configs:
        cfgs = [cfgs[int(i)] for i in a.configs.split(",")]
    folds = [int(i) for i in a.folds.split(",")] if a.folds else range(K)
    for cfg in cfgs:
        for k in folds:
            base = run_path(cfg, k)
            if ext(base, ".json").exists():
                continue
            t0 = time.perf_counter()
            held, stop = d.fold == k, d.fold == (k + 1) % K
            trn = ~(held | stop)
            if a.family in DL:
                rtr, rst, rhe = std_rr(d.RR[trn], d.RR[stop], d.RR[held])
                model, best_ep, status = train_dl(cfg, d.X[trn], rtr, d.y[trn], SEARCH_SEED, cfg["epochs"],
                                                  es=(d.X[stop], rst, d.y[stop]))
                p = predict(model, d.X[held], rhe)
                extra = {"best_epoch": best_ep, "n_params": count_params(model), "status": status}
            else:
                mdl = make_sk(cfg, SEARCH_SEED).fit(d.F[trn], d.y[trn])
                p = mdl.predict_proba(d.F[held])[:, 1]
                extra = {"best_epoch": None, "n_params": None, "status": "ok"}
            yh = d.y[held]
            res = {"family": a.family, "config": cfg["name"], "fold": k, "seed": SEARCH_SEED,
                   "n_held": int(held.sum()), "pr_auc": float(average_precision_score(yh, p)),
                   "roc_auc": float(roc_auc_score(yh, p)), "time_s": time.perf_counter() - t0, **extra}
            np.save(ext(base, ".npy"), p.astype(np.float32))
            ext(base, ".json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            print(f"[{a.family}] {cfg['name']} fold{k} PR-AUC={res['pr_auc']:.4f} ep={res['best_epoch']} "
                  f"t={res['time_s']:.0f}s", flush=True)


def table_for(family, d):
    rows = []
    for cfg in GRIDS[family]:
        rs = [json.loads(ext(run_path(cfg, k), ".json").read_text(encoding="utf-8"))
              for k in range(K) if ext(run_path(cfg, k), ".json").exists()]
        if len(rs) < K:
            rows.append({"family": family, "config": cfg["name"], "folds_done": len(rs)})
            continue
        oof = np.zeros(len(d.y), dtype=np.float32)
        for k in range(K):
            oof[d.fold == k] = np.load(ext(run_path(cfg, k), ".npy"))
        pr = [r["pr_auc"] for r in rs]
        rows.append({"family": family, "config": cfg["name"], "folds_done": K,
                     "cv_pr_auc_mean": float(np.mean(pr)), "cv_pr_auc_std": float(np.std(pr, ddof=1)),
                     "oof_pr_auc": float(average_precision_score(d.y, oof)),
                     "oof_roc_auc": float(roc_auc_score(d.y, oof)),
                     "median_best_epoch": None if rs[0]["best_epoch"] is None else float(np.median([r["best_epoch"] for r in rs])),
                     "time_s_total": float(sum(r["time_s"] for r in rs)),
                     "fold_pr_auc": [round(x, 4) for x in pr]})
    return pd.DataFrame(rows), None


def cmd_select(a, d):
    OUT.mkdir(parents=True, exist_ok=True)
    sel, tabs = {}, []
    for fam in FAMILIES:
        if a.family and fam != a.family:
            continue
        df, _ = table_for(fam, d)
        tabs.append(df)
        ok = df[df.folds_done == K]
        if ok.empty:
            continue
        best = ok.sort_values("cv_pr_auc_mean", ascending=False).iloc[0]
        cfg = next(c for c in GRIDS[fam] if c["name"] == best.config)
        oof = np.zeros(len(d.y), dtype=np.float32)
        for k in range(K):
            oof[d.fold == k] = np.load(ext(run_path(cfg, k), ".npy"))
        thr = ev.select_threshold(d.y, oof)
        sel[fam] = {"config": cfg, "cv_pr_auc_mean": float(best.cv_pr_auc_mean), "cv_pr_auc_std": float(best.cv_pr_auc_std),
                    "oof_pr_auc": float(best.oof_pr_auc), "median_best_epoch": best.median_best_epoch,
                    "oof_threshold_max_f1": thr, "oof_metrics_at_threshold": ev.compute_metrics(d.y, oof, thr),
                    "configs_complete": int(len(ok)), "configs_total": len(GRIDS[fam])}
        print(f"[select] {fam}: {best.config} cv={best.cv_pr_auc_mean:.4f}±{best.cv_pr_auc_std:.4f} thr={thr:.4f}", flush=True)
    path = OUT / (f"selection_{a.family}.json" if a.family else "selection.json")
    path.write_text(json.dumps(sel, indent=2, default=str), encoding="utf-8")
    pd.concat(tabs).to_csv(OUT / (f"search_table_{a.family}.csv" if a.family else "search_table.csv"), index=False)


def cmd_final(a, d):
    FINAL.mkdir(parents=True, exist_ok=True)
    fam = a.family
    sel = json.loads((OUT / f"selection_{fam}.json").read_text(encoding="utf-8"))[fam]
    cfg, thr = sel["config"], sel["oof_threshold_max_f1"]
    epochs = max(5, int(round(sel["median_best_epoch"]))) if fam in DL else None
    for seed in ([int(i) for i in a.seeds.split(",")] if a.seeds else FINAL_SEEDS):
        path = FINAL / f"{fam}__seed{seed}.json"
        if path.exists():
            continue                                  # le test n'est jamais relu pour une (famille, seed) déjà évaluée
        if fam == "lr" and seed != FINAL_SEEDS[0]:
            continue                                  # LR déterministe
        t0 = time.perf_counter()
        if fam in DL:
            rtr, rte = std_rr(d.RR, d.RRt)
            model, _, status = train_dl(cfg, d.X, rtr, d.y, seed, epochs, es=None)
            p = predict(model, d.Xt, rte)
            extra = {"epochs_fixed": epochs, "n_params": count_params(model), "status": status}
        else:
            mdl = make_sk(cfg, seed).fit(d.F, d.y)
            p = mdl.predict_proba(d.Ft)[:, 1]
            extra = {"status": "ok"}
        res = {"family": fam, "config": cfg["name"], "seed": seed, "threshold_oof_max_f1": thr,
               "test_at_oof_thr": ev.compute_metrics(d.yt, p, thr), "test_at_0.5": ev.compute_metrics(d.yt, p, 0.5),
               "train_time_s": time.perf_counter() - t0, **extra}
        by_rec, _ = ev.error_tables(d.rect, d.symt, d.yt, p >= thr)
        by_rec.to_csv(FINAL / f"{fam}__seed{seed}_errors_by_record.csv", index=False)
        np.save(FINAL / f"{fam}__seed{seed}_test_probs.npy", p.astype(np.float32))
        path.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"[final] {fam} seed{seed} test PR-AUC={res['test_at_oof_thr']['pr_auc']:.4f} "
              f"F1@thr={res['test_at_oof_thr']['f1']:.4f}", flush=True)


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def cmd_report(a, d):
    TABLES.mkdir(parents=True, exist_ok=True)
    tabs = []
    for fam in FAMILIES:
        df, _ = table_for(fam, d)
        tabs.append(df)
    s = pd.concat(tabs)
    s["fold_pr_auc"] = s["fold_pr_auc"].astype(str)
    s.to_csv(TABLES / "E2_search_cv.csv", index=False)
    (TABLES / "E2_search_cv.md").write_text(md(s), encoding="utf-8")
    rows = []
    for fam in FAMILIES:
        rs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(FINAL.glob(f"{fam}__seed*.json"))]
        if not rs:
            continue
        g = lambda key, m: np.array([r[key][m] for r in rs])  # noqa: E731
        row = {"family": fam, "config": rs[0]["config"], "n_seeds": len(rs), "threshold_oof": rs[0]["threshold_oof_max_f1"]}
        for m in ("pr_auc", "roc_auc", "f1", "sensitivity", "specificity", "precision", "mcc"):
            v = g("test_at_oof_thr", m)
            row[f"test_{m}_mean"], row[f"test_{m}_std"] = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
        v = g("test_at_0.5", "f1")
        row["test_f1_at_0.5_mean"] = float(v.mean())
        rows.append(row)
    if rows:
        f = pd.DataFrame(rows)
        f.to_csv(TABLES / "E2_final_test_mean_std.csv", index=False)
        (TABLES / "E2_final_test_mean_std.md").write_text(md(f), encoding="utf-8")
        print(md(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["search", "select", "final", "report", "folds"])
    ap.add_argument("--family", choices=FAMILIES)
    ap.add_argument("--configs", help="indices de configurations (ex. 0,1,2) dans la grille de la famille")
    ap.add_argument("--folds", help="search : plis à exécuter (ex. 3,4), pour répartir sur plusieurs processus")
    ap.add_argument("--seeds", help="final : seeds à exécuter (ex. 45,46), pour répartir sur plusieurs processus")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    d = Data()
    if a.cmd == "folds":
        print(json.dumps(json.loads((OUT / "folds.json").read_text(encoding="utf-8"))["folds"], indent=2))
        return
    if a.cmd in ("search", "final") and not a.family:
        ap.error("--family requis")
    {"search": cmd_search, "select": cmd_select, "final": cmd_final, "report": cmd_report}[a.cmd](a, d)


if __name__ == "__main__":
    main()

