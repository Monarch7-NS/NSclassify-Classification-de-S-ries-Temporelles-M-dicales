"""Génère les tableaux Markdown/CSV à partir de results/experiments/index.json. Aucune valeur n'est saisie à la main.

Usage (depuis PROJET/) :  python -m scripts.make_tables
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_tracker import EXP_DIR, load_index  # noqa: E402

OUT = EXP_DIR.parent / "tables"
VAL = ["val_pr_auc", "val_roc_auc", "val_f1", "val_sensitivity", "val_specificity", "val_precision", "val_mcc"]
TEST = [c.replace("val_", "test_") for c in VAL]
COST = ["n_params", "model_bytes", "epoch_time_s", "train_time_s", "latency_ms_single", "throughput_beats_s"]


def save(df, name, floatfmt=4):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(
            f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) and not pd.isna(v) else
            ("" if pd.isna(v) else str(v)) for v in r) + " |")
    (OUT / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    idx = load_index()
    if not idx:
        print("index vide")
        return
    df = pd.DataFrame(idx.values())
    for c in VAL + TEST + COST + ["lr", "weight_decay", "best_epoch", "epochs_run", "overfit_gap_last_epoch"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    base = df[df.stage == "baseline"]
    if len(base):
        save(base[["family", "arch", "id"] + VAL + ["best_epoch", "epochs_run"]].sort_values("family"), "T01_baseline_validation")

    fac = df[df.stage.isin(["baseline", "factors", "factorial"])]
    for fam in ("mlp", "cnn", "cnn_lstm", "lstm"):
        sub = fac[fac.family == fam].sort_values("val_pr_auc", ascending=False)
        if len(sub):
            save(sub[["stage", "factor", "id"] + VAL + ["best_epoch", "epochs_run", "overfit_gap_last_epoch", "status"]],
                 f"T02_factors_{fam}")

    # effet de chaque facteur = écart de PR-AUC val à la référence de la même famille (une seule exécution : pas d'intervalle)
    rows = []
    for fam, g in df[df.stage.isin(["baseline", "factors"])].groupby("family"):
        ref = g[g.stage == "baseline"]
        if not len(ref):
            continue
        r0 = float(ref.val_pr_auc.iloc[0])
        for _, r in g[g.stage == "factors"].iterrows():
            rows.append({"family": fam, "factor": r.factor, "val_pr_auc": r.val_pr_auc, "delta_vs_baseline": r.val_pr_auc - r0,
                         "status": r.status})
    if rows:
        save(pd.DataFrame(rows).sort_values(["family", "delta_vs_baseline"], ascending=[True, False]), "T03_factor_effects")

    fin = df[(df.stage == "final") & (df.status == "ok")]
    if len(fin):
        fin = fin.assign(config=fin.family + ":" + fin.arch.fillna("") + ":" + fin.factor.str.replace("final<-", "", regex=False))
        agg = []
        for cfgname, g in fin.groupby("config"):
            row = {"model": g.family.iloc[0], "arch": g.arch.iloc[0], "selected_from": cfgname.split(":")[-1], "n_seeds": len(g)}
            for c in TEST + ["val_pr_auc", "val_f1"]:
                if c in g:
                    row[c + "_mean"], row[c + "_std"] = g[c].mean(), g[c].std(ddof=1) if len(g) > 1 else np.nan
            agg.append(row)
        save(pd.DataFrame(agg).sort_values("test_pr_auc_mean", ascending=False), "T04_final_test_mean_std")
        cost = fin.groupby("family")[COST].median().reset_index()
        save(cost, "T05_computational_cost")
        # tableau par seuil à partir de metrics.json (test, seuils figés sur validation)
        trs = []
        for _, r in fin.iterrows():
            m = json.loads((EXP_DIR / r.id / "metrics.json").read_text(encoding="utf-8"))
            for thr_name, mm in m["test_metrics"].items():
                trs.append({"id": r.id, "family": r.family, "seed": r.seed, "threshold_rule": thr_name,
                            **{k: mm[k] for k in ("threshold", "sensitivity", "specificity", "precision", "f1", "fp", "fn")}})
        save(pd.DataFrame(trs).groupby(["family", "threshold_rule"]).mean(numeric_only=True).reset_index(), "T06_threshold_study_test")

    fail = df[df.status != "ok"]
    if len(fail):
        save(fail[["id", "stage", "family", "factor", "status"]], "T07_failed_or_diverged")
    print("tables écrites dans", OUT)


if __name__ == "__main__":
    main()
