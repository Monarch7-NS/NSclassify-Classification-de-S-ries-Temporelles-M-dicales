# EXPERIMENT_PLAN — Étude expérimentale MIT-BIH (normal vs anormal)

## 1. Audit du dépôt (état avant ce travail)
- Legacy de Meriem : `scripts/preprocess.py` (npz train/val/test, mapping « 5 classes → N vs non-N », 201 en train), `scripts/essai.py`, `README.md`, `requirements.txt`, `setup_env.sh` (référence un `scripts/download_data.py` absent), `notebooks/demo.ipynb` (vide).
- Premier pipeline LR/RF/CNN sur ces npz : `src/legacy_data.py`, `src/train_legacy.py`, résultats `results/{lr,rf,cnn}/`.
- Données brutes complètes dans `data/mitdb` (48 enregistrements).
- Environnement : Python 3.14, torch 2.14 CPU, sklearn 1.9, 16 threads, 27,7 Go. Pas de GPU, pas de PyYAML.
- Ce qui manquait par rapport au protocole demandé : mapping strict, split documenté, 201/202 ensemble, fenêtre 216, prétraitements comparés, MLP, CNN-LSTM, pertes alternatives, traçage d'expériences, tests anti-fuite.

## 2. Conflits entre protocoles et résolution
| Point | Meriem (legacy) | Protocole retenu |
|---|---|---|
| Mapping | regroupement MIT-BIH → N vs autres | `strict_v1` : N=0 ; L R B A a J S V r F e j n E = 1 ; `/ f Q ?` exclus |
| Split | DS1/DS2 avec 201 en train | par groupes patients ; 201 avec 202 en test |
| Fenêtre | autre longueur | 360 (référence, alignée sur Meriem) ; ablations 216 et 280 |
| Normalisation | variable | z-score par battement (P1) ; P0, P2, P3 en ablation |
Décision : protocole retenu par défaut. Legacy conservé tel quel.

## 3. Protocole
Valeurs dans `configs/experiment.json`. Splits dans `configs/splits.json` (générés par `scripts/build_dataset.py`).

Contrôles avant tout entraînement (réalisés : `python -m pytest tests -q`, 59 tests) : enregistrements disjoints, groupes patients disjoints, 201/202 ensemble, pas de stimulateur, mapping, fit des scalers refusé hors train, augmentation refusée hors train, formes de tous les modèles, pertes, centrage du pic R.

## 4. Modèles
| Famille | Rôle | Réglage |
|---|---|---|
| MLP | profond, focus | presets mlp_a/b/c + facteurs |
| CNN 1D | profond, focus | presets cnn_s/m, fcn + facteurs |
| CNN-LSTM | profond, focus | presets cnnlstm_s/m, cnn_bilstm + facteurs |
| LR | référence | grille 16 points (C, L1/L2, class_weight) sur PR-AUC val |
| RF | référence | grille 18 points (profondeur, feuille min., class_weight) |

## 5. Matrice d'expériences
Référence profonde : P1, fenêtre 360, waveform seule, BCE pondérée, AdamW lr 1e-3 wd 1e-4, batch 128, 30 époques, patience 5.
Étape `factors` : un facteur à la fois (liste complète : `configs/experiment.json > factors`).

| Axe | Valeurs | Familles |
|---|---|---|
| Prétraitement | P0, P2, P3 | toutes |
| Fenêtre | 216, 280 | toutes |
| Contexte RR | oui | toutes |
| Perte | BCE, focal γ=1, γ=2, MSE (inadaptée) | toutes |
| Optimiseur | Adam, SGD+Nesterov | toutes |
| Taux d'apprentissage | 1e-2, 3e-4 | toutes |
| Weight decay | 0, 1e-3 | toutes |
| Batch | 32, 64 | toutes |
| Scheduler | plateau, cosine | toutes |
| Augmentation | légère | toutes |
| Échantillonnage | pondéré | toutes |
| Architecture | mlp_b/c ; cnn_s/fcn ; cnnlstm_m/cnn_bilstm | par famille |
| Activation, dropout, BN | voir config | MLP, CNN, CNN-LSTM |
| Noyau, blocs, pooling, tête | 3, 7, multi-échelle, 2 blocs, stride, flatten | CNN, CNN-LSTM |
| LSTM | hidden 32/128, 2 couches, rnn_dropout, agrégation mean/attention, clip | CNN-LSTM |

Nombre de configurations : 20 communes + 7 (MLP) + 10 (CNN) + 17 (CNN-LSTM) + baselines. Commande de dénombrement : `python -m src.run_experiments --stage factors --dry-run`.

Étape `factorial` : sur la meilleure configuration par famille, perte {BCE, BCE pondérée, focal γ=2} × augmentation {non, oui}.
Étape `final` : top-2 par famille × seeds 42/43/44 (test évalué une fois), LR et RF sur 3 seeds.

## 6. Priorités si le temps manque
1. baseline (5 modèles) ; 2. perte, RR, prétraitement, optimiseur/lr ; 3. architecture ; 4. fenêtre, batch, scheduler ; 5. augmentation, échantillonnage ; 6. facteurs propres à CNN-LSTM ; 7. factorial ; 8. final.

## 7. Sorties
- `results/experiments/<id>/` (config, métriques val, historique, figures, `run_info.json`) ; `results/experiments/index.{json,csv}`.
- `results/tables/T01…T07` générés par `python -m scripts.make_tables` : baseline validation ; facteurs par famille ; effets des facteurs ; comparaison finale test moyenne ± écart-type ; coût de calcul ; seuils ; échecs.
- Analyse d'erreurs (par enregistrement, par symbole), robustesse au bruit, attribution : étape `final` uniquement.

## 8. Limites connues
- Un run par configuration à l'étape `factors` (seed 42) : les écarts inférieurs à la variabilité entre seeds ne sont pas interprétables ; la variabilité est mesurée à l'étape `final` seulement.
- Validation de 5 patients : bruit de sélection non négligeable (hétérogénéité inter-patients forte).
- Pas de validation croisée par patient (coût) ; à envisager si le temps le permet.
- Filtrage non causal.
