"""Orchestrateur par étapes. Reprise automatique : une configuration déjà terminée (status ok) n'est pas ré-exécutée.

Usage (depuis PROJET/) :
  python -m src.run_experiments --stage baseline
  python -m src.run_experiments --stage factors --family cnn --limit 5
  python -m src.run_experiments --stage factors --dry-run          # liste la matrice sans entraîner
  python -m src.run_experiments --stage select                     # écrit results/selection.json
  python -m src.run_experiments --stage factorial
  python -m src.run_experiments --stage final                      # seule étape qui évalue le test
"""
import argparse
import copy
import json
import sys

from . import experiment_tracker as tr
from .data_processing import load_config

DEEP = ("mlp", "cnn", "cnn_lstm")
SKLEARN = ("lr", "rf")
ALL = DEEP + SKLEARN


def set_path(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = copy.deepcopy(value)


def base_config(cfg, family, stage, seed=None, epochs=None, factor="baseline"):
    if family in DEEP:
        c = copy.deepcopy(cfg["defaults"]["deep"])
        spec = copy.deepcopy(cfg["defaults"]["models"][family])
        c["clip_grad"] = spec.pop("clip_grad", c["clip_grad"])
        c["model"] = {"family": family, **spec}
    else:
        c = {"preprocess": cfg["defaults"]["deep"]["preprocess"], "window": cfg["defaults"]["deep"]["window"],
             "use_rr": True, "model": {"family": family, **copy.deepcopy(cfg["defaults"]["sklearn"][family])}}
    c.update(stage=stage, seed=seed if seed is not None else cfg["seeds"]["default"], factor=factor)
    if epochs and family in DEEP:
        c["epochs"] = epochs
    return c


def plan_baseline(cfg, families, epochs):
    return [base_config(cfg, f, "baseline", epochs=epochs) for f in families]


def plan_factors(cfg, families, epochs):
    out = []
    for f in families:
        if f not in DEEP:
            continue
        for fac in cfg["factors"]["common"] + cfg["factors"][f]:
            c = base_config(cfg, f, "factors", epochs=epochs, factor=fac["label"])
            for k, v in fac["set"].items():
                set_path(c, k, v)
            out.append(c)
    return out


def ranked(stages, families, k):
    """Meilleures configurations par famille selon val_pr_auc (critère déclaré à l'avance)."""
    idx = tr.load_index()
    chosen = []
    for f in families:
        rows = [r for r in idx.values() if r["family"] == f and r.get("stage") in stages
                and r.get("status") == "ok" and r.get("val_pr_auc") not in (None, "")]
        rows.sort(key=lambda r: -float(r["val_pr_auc"]))
        for r in rows[:k]:
            c = json.loads((tr.EXP_DIR / r["id"] / "config.json").read_text(encoding="utf-8"))
            c["_selected_from"] = r["id"]
            c["_val_pr_auc"] = float(r["val_pr_auc"])
            chosen.append(c)
    return chosen


def plan_factorial(cfg, families, epochs):
    out = []
    for c in ranked(("baseline", "factors"), [f for f in families if f in DEEP], 1):
        for loss in cfg["factorial"]["loss"]:
            for aug in cfg["factorial"]["augment"]:
                n = copy.deepcopy(c)
                for k in ("_selected_from", "_val_pr_auc"):
                    n.pop(k, None)
                n.update(stage="factorial", loss=loss, augment=aug,
                         factor=f"loss={loss['name']}{loss.get('gamma', '')},augment={aug}")
                out.append(n)
    return out


def plan_final(cfg, families, epochs):
    out = []
    for c in ranked(("baseline", "factors", "factorial"), [f for f in families if f in DEEP], cfg["selection"]["top_k_per_family"]):
        for seed in cfg["seeds"]["final"]:
            n = copy.deepcopy(c)
            src = n.pop("_selected_from")
            n.pop("_val_pr_auc")
            n.update(stage="final", seed=seed, factor=f"final<-{src}")
            out.append(n)
    for f in [x for x in families if x in SKLEARN]:
        for seed in cfg["seeds"]["final"]:
            out.append(base_config(cfg, f, "final", seed=seed, factor="final"))
    return out


PLANS = {"baseline": plan_baseline, "factors": plan_factors, "factorial": plan_factorial, "final": plan_final}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(PLANS) + ["select"])
    ap.add_argument("--family", nargs="*", default=list(ALL), choices=ALL)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config()

    if args.stage == "select":
        sel = ranked(("baseline", "factors", "factorial"), [f for f in args.family if f in DEEP], cfg["selection"]["top_k_per_family"])
        (tr.EXP_DIR.parent / "selection.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
        for c in sel:
            print(c["model"]["family"], c["_selected_from"], round(c["_val_pr_auc"], 4), c["factor"])
        return

    plan = PLANS[args.stage](cfg, args.family, args.epochs)[args.skip:]
    if args.limit:
        plan = plan[:args.limit]
    done = tr.load_index()
    print(f"{len(plan)} configurations, stage={args.stage}", flush=True)
    if args.dry_run:
        for c in plan:
            i = tr.config_id(c)
            print(f"{i:28s} {'done' if done.get(i, {}).get('status') == 'ok' else 'todo':5s} {c['factor']}")
        return

    from .train import run_experiment  # import tardif : évite de charger torch pour un simple listing
    for n, c in enumerate(plan, 1):
        r = run_experiment(c, force=args.force)
        vm = r.get("val_metrics", {}).get(r.get("primary_threshold", "max_f1"), {})
        print(f"[{n}/{len(plan)}] {r['id']} {c['factor']:32s} {r.get('status')} "
              f"val PR-AUC={vm.get('pr_auc', float('nan')):.4f} F1={vm.get('f1', float('nan')):.4f} "
              f"t={r.get('train_time_s', 0):.0f}s", flush=True)
        if r.get("status") == "failed":
            print(r["error"].splitlines()[-1], flush=True)


if __name__ == "__main__":
    sys.exit(main())
