# Analyse des résultats finaux (strict_v1, fenêtre par défaut 360)

Sources : `results/tables/T04_final_test_mean_std.md`, `T05`, `T06`, `results/selection.json`,
`results/figures/fig11–fig18`, comptage des symboles par record (cache `data/processed/mitdb_strict_v1_{train,val,test}.npz`).
Test évalué une seule fois, 3 seeds (42, 43, 44) par famille ; seuil figé sur la validation.

## Résultats test (moyenne ± écart-type sur 3 seeds ; hasard PR-AUC = prévalence = 0.263)

| famille | config sélectionnée (fenêtre) | PR-AUC | ROC-AUC | F1 | sens. | spéc. | MCC |
|---|---|---|---|---|---|---|---|
| cnn | factors-cnn-538ac7f9 (216) | 0.754 ± 0.029 | 0.896 | 0.543 | 0.985 | 0.414 | 0.380 |
| mlp | factors-mlp-a73da86a (360, batch 64) | 0.602 ± 0.033 | 0.802 | 0.540 | 0.609 | 0.768 | 0.354 |
| cnn_lstm | factors-cnn_lstm-17984837 (360, layers 2, rnn_dropout 0.4) | 0.563 ± 0.145 | 0.727 | 0.466 | 0.657 | 0.631 | 0.277 |
| lr | grille interne | 0.542 ± 0 | 0.701 | 0.388 | 0.320 | 0.883 | 0.237 |
| rf | grille interne | 0.514 ± 0.004 | 0.650 | 0.386 | 0.409 | 0.749 | 0.156 |

## Déductions établies (chiffres vérifiés)

1. Classement sur métriques indépendantes du seuil : CNN > MLP > CNN-LSTM ≈ LR > RF. Le CNN domine la courbe PR sur
   rappel > 0.2 (fig16, seed 42 : AP 0.783 / AUC 0.906). Le CNN-LSTM n'est pas supérieur au CNN : l'hypothèse
   « la mémoire récurrente ajoute de la valeur » n'est pas soutenue ici (PR-AUC seeds 0.396 / 0.635 / 0.656, écart-type 0.145).
2. Biais de sélection (winner's curse) : PR-AUC val de la config choisie (1 seed) vs moyenne 3 seeds :
   MLP 0.793 -> 0.727, CNN 0.728 -> 0.615, CNN-LSTM 0.731 -> 0.609. Écart 0.07–0.11. Le classement val (MLP > CNN ≈ CNN-LSTM)
   ne prédit pas le classement test (CNN en tête) : la validation (5 patients) ne discrimine pas les modèles.
3. Transfert de seuil raté : la règle « max-F1 sur val » est moins bonne que le seuil 0.5 sur test pour CNN
   (F1 0.543 vs 0.653), CNN-LSTM (0.466 vs 0.493), LR (0.388 vs 0.395), MLP égal, RF seul gagnant (0.386 vs 0.370) (T06).
   CNN : seuil moyen 0.057, sensibilité 0.985, spécificité 0.414 -> proche du classifieur « tout anormal » (F1 trivial = 0.417).
   Le défaut du CNN est la calibration/le seuil, pas le pouvoir de classement.
4. Hétérogénéité par patient (fig17) : les erreurs sont concentrées par record. Les FN massifs sont sur les records
   à bloc de branche : 111 (L 2121/2122), 212 (R 1824), 214 (L 2001), 231 (R 1252), 232 (R 396 + A 1381). Les blocs de branche L/R
   représentent 55.5 % des anormaux du train mais proviennent de seulement 2 records (109 L, 118 R) ; 55.9 % des anormaux du test
   (4–5 records) ; 73.6 % des anormaux de la validation (124 R, 207 L). Le CNN récupère ces records (FN ≈ 0) mais au prix de
   FP ≈ 1.0 sur 105, 117, 121, 233 et ≈ 0.9–0.99 sur 210, 213, 219, 222, 228.
   Hypothèse forte (non encore testée par modèle) : la difficulté centrale est la généralisation inter-patients sur 2 patients
   de bloc de branche en train, pas le choix de modèle.
5. Bruit du plan factoriel (fig11, fig13) : une seule seed ; écart-type inter-seeds sur val = 0.06–0.12 ; la majorité des
   effets de facteurs (|Δ| < 0.1) est indiscernable du bruit. La distribution CNN est bimodale (~0.45 vs ~0.72) : régime
   « généralise aux BBB de la validation » ou non.
6. Le CNN final utilise window=216, MLP et CNN-LSTM utilisent 360. Le tableau T04 compare donc des fenêtres différentes ;
   aucune conclusion sur « la fenêtre 360 » n'est permise.
7. LR/RF utilisent 14 descripteurs morphologiques + RR (src/features.py), pas le signal brut : la comparaison est
   « DL signal brut » vs « ML descripteurs ». LR est déterministe : écart-type 0 non informatif.
8. Coût (T05) : CNN 52 161 params, 6.7 s/époque, 0.38 ms/battement ; CNN-LSTM 77 313, 23.4 s/époque, 1.09 ms ;
   MLP 54 529, 1.27 s/époque, 0.05 ms ; RF 11.9 Mo, 41 ms/battement.

## Bras RR (results_rr/, post-hoc exploratoire ; 3 RR [rr_prev, rr_next, rr_ratio_prev] pour tous les réseaux)
Test, moyenne ± écart-type sur 3 seeds (results_rr/tables/T04) :
- MLP (P2, bce_pw) PR-AUC 0.6030±0.0093, F1 0.4231 ; CNN (bce_pw, sans aug.) 0.5765±0.0214 ; CNN (bce_pw+aug.) 0.5439±0.0143 ;
  LR 0.5418 ; LSTM Meriem 0.5335±0.0168 ; RF 0.5139±0.0041 ; CNN-LSTM 0.4884±0.0538.
- LSTM de Meriem : test recalculé depuis les checkpoints = valeurs de son train.log (0.5526/0.5213/0.5266 par seed ; seuils identiques).
- Écart val->test important (CNN 0.71 val vs 0.58 test) : le gain val de l'augmentation (0.7695 seed 42) n'est pas confirmé (3 seeds : 0.7876/0.6923/0.6136).
- ATTENTION : le CNN-LSTM final a use_rr=false (retenu par l'étape factors, val 0.6419 vs 0.6124 avec RR) et le MLP a P2 :
  le bras n'est pas homogène pour ces deux familles. CNN-LSTM : spécificité test 0.31, sensibilité 0.84 (seuil val mal transféré).

- Erreurs par record (bras RR, moyenne 3 seeds, seuil max-F1 val) : les 5 records à bloc de branche (111, 212, 214, 231, 232 ;
  9 237 des 13 588 anormaux test = 68 %) concentrent 7 022-8 717 des FN du CNN/MLP/LR/LSTM/RF (MLP : 8 717 / 9 601 ; rappel
  MLP ~6 % sur ces 5 records, ~80 % ailleurs). Le CNN-LSTM récupère ces records (FN 2 167) au prix de 26 096 FP (spéc. 0.31).
- Comparaison avec le bras sans RR (section précédente) : CNN 0.754 (sans RR, fenêtre 216) vs 0.577 (RR) ; MLP 0.602 vs 0.603 ;
  CNN-LSTM 0.563 vs 0.488. Configs/fenêtres différentes et distribution CNN bimodale : ne pas conclure que le RR nuit, mais
  il n'aide pas.

## Limites / suites (non faites)
- n = 3 seeds mesure la variance d'initialisation, pas la variance d'échantillonnage de patients.
- Remède principal : validation croisée groupée par patient (GroupKFold) ou bootstrap par patient ; métriques par record ;
  analyse de sensibilité sans L/R. Ne pas re-découper le split après lecture du test.
- fig14 trace la PR-AUC val de la baseline, pas le résultat final : à régénérer avec la PR-AUC test finale.
- Non fait : LSTM seul, facteurs LR/RF, ResNet.
