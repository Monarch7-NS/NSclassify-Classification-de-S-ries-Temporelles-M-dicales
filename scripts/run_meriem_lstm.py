"""Exécute le LSTM de Meriem (scripts/meriem_lstm_original.py, code non modifié) sur le split strict_v1 du projet.

  python -m scripts.run_meriem_lstm prepare   # écrit results/meriem_lstm/data/mitdb_{train,val,test}.npz au format attendu par son script
  python -m scripts.run_meriem_lstm train     # lance son script tel quel dans results/meriem_lstm/ (journal horodaté : train.log)
  python -m scripts.run_meriem_lstm integrate # reconstruit probabilités/métriques depuis ses checkpoints et les ajoute à results/experiments
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SANDBOX = ROOT / "results" / "meriem_lstm"
ORIGINAL = ROOT / "scripts" / "meriem_lstm_original.py"
RR_COLS = [0, 1, 3]  # rr_prev, rr_next, rr_ratio_prev (standardisés sur train) : ses 3 entrées RR


def prepare():
    from src.datasets import prepare as prep_data
    p = prep_data("P1", 360)
    (SANDBOX / "data").mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        np.savez(SANDBOX / "data" / f"mitdb_{s}.npz", X=p.X[s].astype(np.float32),
                 RR=p.RR[s][:, RR_COLS].astype(np.float32), y=p.y[s].astype(np.int64))
        print(s, p.X[s].shape, "prévalence", round(float(p.y[s].mean()), 4))


def train():
    import os
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore"}
    proc = subprocess.Popen([sys.executable, "-u", str(ORIGINAL)], cwd=SANDBOX, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    t0 = time.time()
    with open(SANDBOX / "train.log", "a", encoding="utf-8") as log:
        for line in proc.stdout:
            line = f"[{time.time() - t0:9.1f}s] {line.rstrip()}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    return proc.wait()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "prepare":
        prepare()
    elif cmd == "train":
        sys.exit(train())
    elif cmd == "integrate":
        from scripts.integrate_meriem_lstm import integrate
        integrate()
    else:
        print(__doc__)
