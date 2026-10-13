"""Télécharge MIT-BIH (et optionnellement PTB-DB) depuis PhysioNet via wfdb."""
from __future__ import annotations

import argparse
from pathlib import Path

import wfdb

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--ptbdb", action="store_true", help="télécharge aussi PTB Diagnostic ECG (≈ 1.7 Go)")
    args = p.parse_args()

    targets = {"mitdb": args.data_dir / "mitdb"}
    if args.ptbdb:
        targets["ptbdb"] = args.data_dir / "ptbdb"

    for db, dest in targets.items():
        if dest.exists() and any(dest.glob("*.hea")):
            print(f"[skip] {db} déjà présent dans {dest}")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[download] {db} -> {dest}")
        wfdb.dl_database(db, dl_dir=str(dest))


if __name__ == "__main__":
    main()
