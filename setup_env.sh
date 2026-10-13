#!/usr/bin/env bash
# Crée l'environnement virtuel, installe les dépendances et télécharge MIT-BIH.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
"$PYTHON" -m venv .venv

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  # Git Bash / Windows
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_data.py

echo "Environnement prêt. Activer avec : source .venv/bin/activate (ou .venv/Scripts/activate sous Windows)"
