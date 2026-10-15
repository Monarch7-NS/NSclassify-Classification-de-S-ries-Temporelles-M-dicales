# Contexte du projet

## Énoncé
DATA931 (Polytech Annecy-Chambéry), projet 3 « Classification de séries temporelles médicales » : classer chaque battement de MIT-BIH comme **normal (0)** ou **anormal (1)**. Binôme : Anas Draoui (NS) et Meriem Khedira. Brief : `src/mit_bih_normal_vs_abnormal_heartbeat_classification_20261013_115241.pdf`.

L'enseignant demande de privilégier les modèles profonds : **MLP, CNN 1D, CNN-LSTM** (focus de l'étude) ; **LR et RF** gardés comme références.

## Données
MIT-BIH Arrhythmia Database 1.0.0 (PhysioNet), 48 enregistrements, 360 Hz, dérivation MLII. Brut : `data/mitdb/` (115.dat avait été tronqué puis re-téléchargé ; contrôle : 1 950 000 octets par .dat).

## Protocole courant (strict_v1) — voir `EXPERIMENTAL_RULES.md`
| Split | Enregistrements | Battements | Anormaux |
|---|---|---|---|
| train | 101 106 108 109 112 114 115 116 118 119 122 208 209 220 223 230 | 36 544 | 8 389 (23,0 %) |
| val | 124 203 205 207 215 | 12 464 | 4 175 (33,5 %) |
| test | DS2 + 201 (100 103 105 111 113 117 121 123 200 201 202 210 212 213 214 219 221 222 228 231 232 233 234) | 51 622 | 13 588 (26,3 %) |

Fenêtres : 360 échantillons stockés (180 avant R, 180 après ; 1,0 s) = fenêtre par défaut ; ablations : 216 = tranche [90:306] (90 avant, 126 après), 280 = tranche [70:350] (110 avant, 170 après). Deux variantes : signal brut ou filtré (Butterworth ordre 4, 0,5–40 Hz, phase nulle, appliqué à l'enregistrement entier). Contexte RR : `rr_prev, rr_next, rr_med` (médiane des ≤ 8 intervalles précédents), `rr_ratio_prev, rr_ratio_next`. Premier et dernier battement de chaque enregistrement ignorés (pas de voisin).

Prétraitements : P0 brut + z-score par battement ; P1 filtré + z-score par battement (référence) ; P2 filtré + standardisation globale train ; P3 filtré + mise à l'échelle robuste (médiane/IQR train).

## Legacy (protocole de Meriem, à ne pas mélanger)
`data/mitdb_{train,val,test}.npz` (`scripts/preprocess.py`), résultats `results/{lr,rf,cnn}/`, code `src/legacy_data.py`, `src/train_legacy.py`. Différences : mapping différent, fenêtre autre, 201 en train. Les chiffres legacy ne sont **pas comparables** aux nouveaux.

## Structure
```
configs/experiment.json   configuration machine-lisible (mapping, splits, facteurs, défauts)
configs/splits.json       partition figée (générée)
data/mitdb/               brut ; data/processed/ cache fenêtres (gitignored)
src/data_processing.py    WFDB -> fenêtres, splits, assertions
src/features.py          RR + morphologie (LR/RF)
src/datasets.py          Preprocessor P0-P3, Prepared, ECGData (augmentation train-only)
src/models.py            LR, RF, MLP, CNN1D, CNNLSTM, presets
src/losses.py            BCE, BCE pondérée, focal, lissage, MSE (ablation)
src/augmentation.py      jitter, échelle, décalage
src/evaluate.py          métriques, seuils, erreurs, coût
src/interpretability.py  coefficients, permutation, gradients intégrés, CAM
src/experiment_tracker.py  ids, run_info, index.json/csv
src/train.py             run_experiment(cfg)
src/run_experiments.py   étapes baseline / factors / factorial / final
scripts/build_dataset.py  construit splits + cache
scripts/make_tables.py    tableaux Markdown/CSV depuis l'index
tests/test_pipeline.py    non-fuite, formes, pertes
results/experiments/<id>/ sorties par expérience
```

## Commandes (depuis `PROJET/`, venv `.venv`)
```
python -m scripts.build_dataset
python -m pytest tests -q
python -m src.run_experiments --stage baseline
python -m src.run_experiments --stage factors --family cnn --dry-run
python -m src.run_experiments --stage factors --family cnn
python -m src.run_experiments --stage factorial
python -m src.run_experiments --stage final
python -m scripts.make_tables
```
