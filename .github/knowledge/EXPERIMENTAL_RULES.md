# Règles expérimentales (non négociables)

1. Le split se fait **par patient** (groupes de sujets) avant toute extraction de fenêtre ou tout ajustement de statistique. 201 et 202 (même sujet) sont dans la même partition.
2. Test = DS2 + 201. Validation = 5 groupes patients tirés de DS1 \ 201, choisis par comptes d'annotations uniquement (`configs/splits.json`). Train = le reste de DS1.
3. Enregistrements à stimulateur (102, 104, 107, 217) exclus. Dérivation MLII uniquement.
4. Étiquettes `strict_v1` : N → 0 ; L R B A a J S V r F e j n E → 1 ; `/ f Q ?` exclus ; symboles non-battement ignorés.
5. Scalers / statistiques globales (P2, P3, standardisation RR, StandardScaler LR) : ajustés sur **train seulement**. P0/P1 (z-score par battement) n'utilisent aucune statistique globale.
6. Augmentation : train uniquement, en ligne (jamais sur val/test) ; imposé par `ECGData.augment_batch`.
7. Early stopping, scheduler, sélection d'hyperparamètres, seuils : validation uniquement. Critère de sélection déclaré à l'avance : **PR-AUC validation**. Seuil de décision : max-F1 sur validation (autres seuils rapportés : 0.5 et sensibilité ≥ 0.90).
8. Test : évalué **une seule fois** par configuration finale (`stage == "final"`), seuils figés ; `run_experiment` refuse de ré-évaluer.
9. Seeds : 42 par défaut ; configurations finales sur 42, 43, 44 ; rapporter moyenne ± écart-type, pas le meilleur seed.
10. Une expérience = un identifiant (`stage-famille-hash`), un dossier `results/experiments/<id>/` avec `config.json`, `metrics.json`, `run_info.json`, `history.csv`, figures. Les échecs et divergences sont enregistrés (`status`).
11. Aucun chiffre dans un rapport sans provenance (id d'expérience).
12. Les features LR/RF sont calculées sur la fenêtre filtrée **avant** normalisation par battement (amplitudes en mV).
13. Un facteur à la fois autour de la configuration de référence (étape `factors`), puis plan factoriel réduit pertes × augmentation sur la meilleure configuration (étape `factorial`). Les interactions non testées ne sont pas affirmées.
