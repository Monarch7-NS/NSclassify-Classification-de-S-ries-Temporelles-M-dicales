# HANDOFF — reprise du travail (état au 2026-10-15, ~01h00)

Projet DATA931 « Classification de séries temporelles médicales » (Anas Draoui "NS" et Meriem Khedira) : classification binaire de battements MIT-BIH (0 = normal, 1 = anormal).
Langue de réponse : français, concis, sans émojis. Persona « Galet » (mentor ML niveau PhD, voir `C:\NS\POLYTECH\DATA931\.github\copilot-instructions.md`). La base de connaissances vit dans `PROJET/.github/knowledge/` (lire d'abord `PROJECT_CONTEXT.md`, `EXPERIMENTAL_RULES.md`, `DECISIONS.md`, `USER_PREFERENCES.md`, `RESULTS_ANALYSIS.md`).

## État en une phrase
Étude 1 (`results/` sans RR, `results_rr/` avec RR) terminée, intégrée et analysée. **Étude 2 (`results_etude2/`, optimisation par CV groupée par patient) lancée le 2026-10-15 ~01h00, en cours.** Rien n'est commité.

## Règles non négociables
- Aucun chiffre inventé : tout nombre vient d'un fichier produit (metrics.json, JSON de `results_etude2/`), cité avec son ID d'expérience (étude 1) ou son nom de config (étude 2).
- Split strict par patient (`configs/splits.json`). Seuil figé avant lecture du test. Sélection uniquement sur validation / CV, **jamais sur le test**.
- Ne pas modifier le code de Meriem (`scripts/meriem_lstm_original.py` = copie verbatim).
- Ne pas commit/push/supprimer sans demander (branche `dev`).
- Ne pas toucher le processus python PID 28276 (hors projet, très gourmand). Les `python -m http.server 8766` (PID 33908, 54612) servent la présentation : ne pas les tuer non plus.
- Ne pas modifier `results/` ni `results_rr/` (terminés, résultats de l'étude 1).

## Environnement
- Windows, PowerShell 5.1. Venv `PROJET\.venv` (Python 3.14, torch CPU, sklearn, numpy, pandas, wfdb). 16 CPU.
- Les terminaux synchrones renvoient souvent « Command produced no output » : vérifier par les fichiers (logs, JSON). `*>` écrit en UTF-16 : utiliser `2>&1 | Out-File -Encoding utf8`.
- Variables : `PYTHONIOENCODING=utf-8`, `PYTHONWARNINGS=ignore`, `EXPERIMENT_CONFIG` (config JSON), `RESULTS_DIR` (étude 1 seulement), `OMP_NUM_THREADS`.

## Protocole strict_v1 (résumé)
- N = 0 ; L,R,B,A,a,J,S,V,r,F,e,j,n,E = 1 ; exclus : `/ f Q ?` ; records rythmés exclus : 102, 104, 107, 217.
- Filtre Butterworth 0,5-40 Hz ordre 4 (zéro-phase), z-score par battement (P1), fenêtre 360 points.
- Split : train 36 544 battements (16 patients), val 12 464 (5 patients : 124, 203, 205, 207, 215), test 51 622 (23 patients, dont 201 et 202). Les « ~40k/10k/50k » de la présentation correspondent à l'ancien split aléatoire, pas à strict_v1.
- Seeds : 42 par défaut ; étude 1 finals [42, 43, 44] ; étude 2 finals [42..46].

## Les trois séries d'expériences
1. **Étude 1a, sans RR** : `results/`. MLP val PR-AUC 0,7926 ; CNN 0,7278 ; CNN-LSTM 0,7307 (config retenues : MLP batch 64, CNN window 216, CNN-LSTM layers 2 / rnn_dropout 0,4).
2. **Étude 1b, avec RR** : `results_rr/`, config `configs/experiment_rr.json` (`use_rr` true, `rr_cols` [0,1,3]). LR/RF : 5 RR + 9 features de morphologie. Chaîne terminée (`results_rr/CHAIN_DONE.txt`), intégration du LSTM de Meriem, tables T01–T06 et figures fig01–fig18 produites. Résultats test (3 seeds, moyenne ± écart-type, `results_rr/tables/T04_final_test_mean_std.csv`) :

   | modèle | PR-AUC test | F1 test | sensibilité | précision |
   |---|---|---|---|---|
   | MLP (`factors-mlp-6ea82177`, P2) | 0,6030 ± 0,0093 | 0,4231 | 0,2934 | 0,7630 |
   | CNN (`factorial-cnn-72349c7d`, sans augmentation) | 0,5765 ± 0,0214 | 0,4579 | 0,4418 | 0,5115 |
   | CNN (`factorial-cnn-6e742995`, avec augmentation) | 0,5439 ± 0,0143 | 0,4257 | 0,3581 | 0,5323 |
   | LR | 0,5418 | 0,3880 | 0,3195 | 0,4938 |
   | LSTM de Meriem | 0,5335 ± 0,0168 | 0,4100 | 0,3254 | 0,5669 |
   | RF | 0,5139 ± 0,0041 | 0,3861 | 0,4089 | 0,3753 |
   | CNN-LSTM (`factors-cnn_lstm-fc4df087`, `use_rr=false`) | 0,4884 ± 0,0538 | 0,4556 | 0,8405 | 0,3139 |

   Constat : but méthodologique atteint (split strict, seuil figé sur val, 3 seeds), performance modeste. Les erreurs sont dominées par les patients à blocs de branche (BBB, battements L/R) : voir `.github/knowledge/RESULTS_ANALYSIS.md`.
   Réserves à répéter dans tout rapport : bras post-hoc ; MLP retenu en P2 (les autres en P1) ; **le CNN-LSTM final n'a pas les RR** (`use_rr=false` sélectionné sur validation) ; le LSTM de Meriem n'a pas d'étape de sélection (entrée baseline = seed 42, val) ; le nombre de threads peut changer les dernières décimales.
3. **Étude 2** (`results_etude2/`) — voir section suivante. Nom : `etude2_cv_groupee` (le nom initial « etude2_hyperparametresV2 » a été remplacé : « V2 » ne dit rien, et la nouveauté est la CV groupée ; le dossier s'appelle `results_etude2`).

## Étude 2 : optimisation par CV groupée par patient (annoncée « optimisation ultérieure »)
**But** : améliorer l'étude 1 sans changer le split de test. Les quatre pistes :
1. Réduire le bruit de sélection : CV groupée par patient (5 plis) sur **train + validation = 21 patients** (au lieu d'une validation de 5 patients), puis 5 seeds en final.
2. Régulariser davantage : weight decay (1e-4, 1e-3, 1e-2), dropout (0,2 à 0,5), modèles plus petits (MLP 64-32, CNN `cnn_s`, tête `gap`, LSTM hidden 32).
3. Régler le seuil à part : seuil max-F1 déterminé sur les **prédictions hors pli (OOF)** de la config retenue, pas sur la validation d'un seul pli.
4. Revoir le CNN-LSTM : lr 3e-4 (au lieu de 1e-3), clip du gradient (1,0 ou 0,5), mean pooling, dropout de tête 0,5, hidden 32 ; **les RR sont fournis** (`use_rr` true) à tous les réseaux.

**Règles anti-triche (à répéter dans le rapport)**
- Sélection uniquement par CV (PR-AUC moyenne des 5 plis) ; le test n'entre dans aucun choix. Le test est lu **une seule fois par (famille, seed)** à l'étape `final` ; le script saute toute (famille, seed) déjà évaluée.
- **Le test a déjà été lu dans l'étude 1** (une fois pour `results/`, une fois pour `results_rr/`). L'étude 2 est donc une **troisième lecture** du même jeu de test : ses résultats sont à présenter comme « optimisation ultérieure » et jamais comme une évaluation vierge. On rapporte **les deux séries** (étude 1 : T04 ; étude 2 : `E2_final_test_mean_std`), même si l'étude 2 est moins bonne.
- Aucune grille n'a été modifiée après avoir vu des résultats de test (aucun résultat de test n'existait au moment de l'écriture des grilles dans `scripts/etude2_cv.py`).

**Conception**
- Plis : `results_etude2/folds.json`, affectation record -> pli déterministe à partir des seuls comptages d'annotations : 4 patients BBB (109, 118, 124, 207) répartis un par pli, puis remplissage glouton par nombre de battements. Pli d'arrêt précoce du pli k = pli (k+1) mod 5 ; entraînement sur les 3 plis restants. RR standardisés par pli.
- Pools : P1, fenêtre 360, perte `bce_pw`, AdamW, batch 128, 30 époques max, patience 5 sur PR-AUC du pli d'arrêt. Seed de recherche 42.
- Grilles (`GRIDS` dans le script) : MLP 6 configs, CNN 7, CNN-LSTM 6, LR 4 (C ∈ {0,01, 0,1, 1, 10}), RF 4 (profondeur 12/20, feuille 1/5). 5 plis chacune.
- Sélection : meilleure PR-AUC moyenne des plis par famille. Rapporté : moyenne ± écart-type des plis, PR-AUC poolée OOF. Écarts inférieurs à l'écart-type des plis = non significatifs.
- Final : modèle entraîné sur les 21 patients, nombre d'époques fixé = médiane des meilleures époques des plis (min 5), 5 seeds (LR : 1 run, déterministe), seuil = max-F1 sur OOF de la config retenue ; test évalué au seuil OOF et à 0,5.
- Pas dans l'étude 2 : le LSTM de Meriem (non ré-ajusté ; on garde son résultat de l'étude 1) ; pas de P2 pour le MLP (P1 pour tous).

**Fichiers**
- Code : `scripts/etude2_cv.py` (sous-commandes `search`, `select`, `final`, `report`, `folds`). Pilote : `results_etude2/run_etude2.ps1` (phases : recherche en parallèle, puis par famille sélection -> final, puis `select` global + `report` + `ETUDE2_DONE.txt`).
- Sorties : `results_etude2/runs/<famille>__<config>__fold<k>__seed42.{json,npy}` ; `selection_<famille>.json` puis `selection.json` ; `search_table*.csv` ; `final/<famille>__seed<s>.json` (+ `_test_probs.npy`, `_errors_by_record.csv`) ; `tables/E2_search_cv.{csv,md}`, `tables/E2_final_test_mean_std.{csv,md}` ; `logs/*.log` ; `ETUDE2_STATUS.txt`, `ETUDE2_DONE.txt`.

**Lancer / reprendre** (reprenable : les runs et finals déjà écrits sont sautés)
```
cd C:\NS\POLYTECH\DATA931\PROJET
Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','results_etude2\run_etude2.ps1'
```
Avant de relancer, vérifier qu'il n'y a pas déjà un pilote vivant : `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select ProcessId,CommandLine` (chercher `etude2_cv.py`) et l'absence de `ETUDE2_DONE.txt`. Un redémarrage de VS Code peut tuer le pilote sans erreur (déjà arrivé le 14/10 à 23h19 pour la chaîne de l'étude 1) ; dans ce cas relancer la commande ci-dessus.
Suivi : `(Get-ChildItem results_etude2\runs -Filter *.json).Count` (total attendu = (6 + 7 + 6 + 4 + 4) configs × 5 plis = 135 runs) ; logs `results_etude2\logs\search_*.out.log`.

**Après `ETUDE2_DONE.txt`**
1. Lire `results_etude2/tables/E2_search_cv.md` et `E2_final_test_mean_std.md`, comparer à T04 (étude 1), vérifier la cohérence (nombre de seeds, seuils OOF, runs `status` ok / diverged).
2. Mettre à jour `.github/knowledge/RESULTS_ANALYSIS.md` (section étude 2), `DECISIONS.md` (entrée `etude2_cv_groupee` : changements de protocole, troisième lecture du test) et ce fichier.
3. Rapport à l'utilisateur en français : résultats avec noms de configs, réserves (troisième lecture du test, protocole modifié : entraînement sur 21 patients, CNN-LSTM avec RR, P1 partout, LSTM de Meriem non ré-ajusté, un seul seed de recherche).

## LSTM de Meriem
- Code d'origine : `scripts/meriem_lstm_original.py` (inchangé). `PureLSTMClassifier` : hidden 128, 2 couches, dropout 0,2 ; dernier état caché + 3 RR -> Linear(131,32) -> ReLU -> Dropout 0,3 -> Linear(32,1) ; BCE avec pos_weight, AdamW, batch 128, 30 époques max, patience 5.
- Runner : `scripts/run_meriem_lstm.py` (`prepare`, `train`, `integrate`) ; intégration : `scripts/integrate_meriem_lstm.py`. Bac à sable : `results/meriem_lstm/`.
- Test (3 seeds, `train.log`) : PR-AUC 0,5335 ± 0,0168 ; F1 0,4100 ± 0,0214. Intégré aux tables T01–T06 de `results_rr`.

## Présentation HTML
- Source : `presentation/template.html` ; sortie : `presentation/index.html` (ne jamais éditer à la main). Régénérer : `python -m scripts.dataset_stats`.
- Contenu : intro normal vs anormal (patients 100 et 208), nettoyage, segmentation (fenêtre 360 + RR), anti-triche (split par patient, De Chazal et al. « adapté »). Les résultats de modèles (tables/figures RR) ne sont pas encore dedans.

## Fichiers clés
`src/{data_processing,datasets,features,models,losses,train,run_experiments,experiment_tracker,evaluate,augmentation}.py`, `configs/{experiment.json,experiment_rr.json,splits.json}`, `scripts/{etude2_cv,make_tables,make_figures,dataset_stats,run_meriem_lstm,integrate_meriem_lstm}.py`, `.github/knowledge/*.md`.
