"""Génère presentation/index.html à partir de presentation/template.html.

Deux jeux de données sont injectés dans le gabarit :
  - DATA (/*__DATA__*/) : statistiques du jeu de données (results/dataset_stats.json,
    produit par scripts.dataset_stats), diapositives 1-2 ;
  - RES  (/*__RES__*/)  : résultats des études 1 et 2, lus dans les fichiers de
    résultats (aucun chiffre saisi à la main), diapositives 3 et suivantes.

Usage (depuis PROJET, venv actif) :  python -m scripts.build_presentation
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRES = ROOT / "presentation"
R1 = ROOT / "results_rr"          # étude 1 (bras RR, 3 seeds)
R1B = ROOT / "results"            # étude 1a (sans RR)
E2 = ROOT / "results_etude2"      # étude 2 (CV groupée)
FAMS = ["lr", "rf", "mlp", "cnn", "cnn_lstm"]
LABEL = {"lr": "LR", "rf": "RF", "mlp": "MLP", "cnn": "CNN 1D", "cnn_lstm": "CNN-LSTM",
         "lstm": "LSTM"}


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def study1():
    """Une ligne par famille : la config de plus forte PR-AUC *validation* (jamais le test)."""
    path = R1 / "tables" / "T04_final_test_mean_std.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    best = {}
    for r in rows:
        fam = r["model"]
        if fam not in best or _f(r["val_pr_auc_mean"]) > _f(best[fam]["val_pr_auc_mean"]):
            best[fam] = r
    out = []
    for fam in ["lr", "rf", "lstm", "mlp", "cnn", "cnn_lstm"]:
        r = best.get(fam)
        if r is None:
            continue
        out.append({
            "k": fam, "label": LABEL[fam], "cfg": r["selected_from"], "n": int(r["n_seeds"]),
            "pr": _f(r["test_pr_auc_mean"]), "sd": _f(r["test_pr_auc_std"]),
            "f1": _f(r["test_f1_mean"]), "sens": _f(r["test_sensitivity_mean"]),
            "spec": _f(r["test_specificity_mean"]), "prec": _f(r["test_precision_mean"]),
            "val": _f(r["val_pr_auc_mean"]),
        })
    return out


def study2():
    """Agrège final/<famille>__seed<s>.json (au seuil OOF) pour les familles déjà terminées."""
    out, errs = [], {}
    for fam in FAMS:
        files = sorted((E2 / "final").glob(f"{fam}__seed*.json"))
        js = [json.loads(p.read_text(encoding="utf-8")) for p in files]
        js = [j for j in js if j.get("status") == "ok"]
        if not js:
            continue
        pr = np.array([j["test_at_oof_thr"]["pr_auc"] for j in js])
        key = lambda k: float(np.mean([j["test_at_oof_thr"][k] for j in js]))
        out.append({
            "k": fam, "label": LABEL[fam], "cfg": js[0]["config"], "n": len(js),
            "pr": float(pr.mean()), "sd": float(pr.std(ddof=1)) if len(pr) > 1 else 0.0,
            "f1": key("f1"), "sens": key("sensitivity"), "spec": key("specificity"),
            "prec": key("precision"), "thr": float(np.mean([j["threshold_oof_max_f1"] for j in js])),
            "params": js[0].get("n_params"),
            "t": float(np.mean([j["train_time_s"] for j in js])),
        })
        errs[fam] = [p.with_name(p.stem + "_errors_by_record.csv") for p in files]
    return out, errs


def bbb_records():
    """Records du test dont plus de la moitié des battements sont L ou R (blocs de branche)."""
    z = np.load(ROOT / "data" / "processed" / "mitdb_strict_v1_test.npz", allow_pickle=True)
    rec, sym, y = z["record_id"], z["symbol"], z["y"]
    bbb = []
    for r in np.unique(rec):
        m = rec == r
        if np.isin(sym[m], ["L", "R"]).mean() > 0.5:
            bbb.append(str(r))
    return bbb, int(len(y)), int(y.sum()), {str(r): int(((rec == r) & (y == 1)).sum()) for r in bbb}


def recall_split(errs, bbb):
    """Rappel (sensibilité) sur les patients à bloc de branche vs les autres, par famille."""
    res = {}
    for fam, paths in errs.items():
        acc = {"b": [0, 0], "o": [0, 0]}
        for p in paths:
            for row in csv.DictReader(p.open(encoding="utf-8")):
                g = "b" if row["record"] in bbb else "o"
                acc[g][0] += int(row["fn"])
                acc[g][1] += int(row["n_abnormal"])
        res[fam] = {"bbb": 1 - acc["b"][0] / acc["b"][1], "other": 1 - acc["o"][0] / acc["o"][1]}
    return res


def study1_errs():
    """Fichiers d'erreurs par record des runs finaux de l'étude 1 (CNN : config sans augmentation)."""
    idx = json.loads((R1 / "experiments" / "index.json").read_text(encoding="utf-8"))
    out = {}
    for name, e in idx.items():
        if e.get("stage") != "final":
            continue
        if e["family"] == "cnn" and e.get("augment") not in (False, 0, "False"):
            continue
        p = R1 / "experiments" / name / "test_errors_by_record.csv"
        if p.exists():
            out.setdefault(e["family"], []).append(p)
    return out


def cost1():
    p = R1 / "tables" / "T05_computational_cost.csv"
    return {r["family"]: {"params": _f(r["n_params"]), "epoch_s": _f(r["epoch_time_s"]),
                          "train_s": _f(r["train_time_s"])}
            for r in csv.DictReader(p.open(encoding="utf-8"))}


def cv_summary():
    out = {}
    for fam in FAMS:
        p = E2 / f"search_table_{fam}.csv"
        if not p.exists():
            continue
        rows = [r for r in csv.DictReader(p.open(encoding="utf-8")) if r["folds_done"] == "5"]
        if not rows:
            continue
        m = [_f(r["cv_pr_auc_mean"]) for r in rows]
        s = [_f(r["cv_pr_auc_std"]) for r in rows]
        b = int(np.argmax(m))
        out[fam] = {"n": len(rows), "best": m[b], "best_cfg": rows[b]["config"], "best_sd": s[b],
                    "worst": min(m), "sd_mean": float(np.mean(s))}
    return out


def counts():
    n1 = len(json.loads((R1 / "experiments" / "index.json").read_text(encoding="utf-8")))
    n1b = len(json.loads((R1B / "experiments" / "index.json").read_text(encoding="utf-8")))
    runs = list((E2 / "runs").glob("*.json"))
    secs = 0.0
    for p in runs:
        try:
            t = float(json.loads(p.read_text(encoding="utf-8")).get("time_s", 0) or 0)
            if t < 3600:   # écarte les runs gonflés par la mise en veille de la machine (~9,5 h)
                secs += t
        except (ValueError, OSError):
            pass
    finals = len(list((E2 / "final").glob("*__seed*.json")))
    return {"exp_study1": n1 + n1b, "runs_e2": len(runs), "finals_e2": finals,
            "hours_e2": secs / 3600.0}


def main():
    s2, errs = study2()
    bbb, n_test, n_abn, abn_by = bbb_records()
    res = {
        "prev": n_abn / n_test, "n_test": n_test, "n_abn": n_abn,
        "s1": study1(), "s2": s2, "cv": cv_summary(), "cnt": counts(),
        "bbb": {"records": bbb, "share": sum(abn_by.values()) / n_abn,
                "recall1": recall_split(study1_errs(), bbb),
                "recall2": recall_split(errs, bbb)},
        "cost1": cost1(),
        "done": (E2 / "ETUDE2_DONE.txt").exists(),
    }
    data = (ROOT / "results" / "dataset_stats.json").read_text(encoding="utf-8")
    html = (PRES / "template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", data).replace("/*__RES__*/null", json.dumps(res, ensure_ascii=False))
    (PRES / "index.html").write_text(html, encoding="utf-8")
    print("presentation/index.html écrit,", len(html), "octets ; étude 2 :",
          [r["k"] for r in s2], "; terminée :", res["done"])


if __name__ == "__main__":
    main()
