"""Construit splits.json (si absent), le cache de fenêtres et le manifeste. Aucun entraînement.

Usage (depuis PROJET/) :  python -m scripts.build_dataset [--force-cache]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_processing import build_cache, load_config, make_splits  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-cache", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    splits = make_splits(cfg)
    print("selection:", splits["selection"])
    for s in ("train", "val", "test"):
        print(s, splits[s])
    manifest = build_cache(cfg, force=args.force_cache)
    for s, m in manifest["splits"].items():
        print(f"{s:5s} n={m['n_beats']:6d} normal={m['n_normal']:6d} abnormal={m['n_abnormal']:6d} "
              f"prev={m['n_abnormal'] / m['n_beats']:.3f}")


if __name__ == "__main__":
    main()
