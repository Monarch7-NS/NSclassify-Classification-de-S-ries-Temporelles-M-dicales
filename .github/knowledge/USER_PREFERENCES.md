# Préférences de NS (à respecter dans ce projet)

## Style de réponse
- Court, impersonnel, direct. Pas d'emoji. Pas de remplissage ni d'introductions.
- Mélange français / anglais accepté ; documents du projet en français, code et noms en anglais.
- Précision avant simplification : formalisme, équations, terminologie exacte ; corriger directement les imprécisions.
- Impatient : agir plutôt que demander, sauf pour les actions destructrices ou irréversibles.

## Discipline de scope
- Faire ce qui est demandé, rien de plus. Ne pas toucher aux fichiers de Meriem (`scripts/essai.py`, `scripts/preprocess.py`, README) sans demande.
- Les anciens résultats (`results/lr`, `results/rf`, `results/cnn`, `data/mitdb_*.npz`) sont du **legacy** : conservés, jamais écrasés.
- Les nouveaux résultats vont dans `results/experiments/`, tableaux dans `results/tables/`.

## Interdits
- Pas de `git commit` / `git push` / suppression de fichiers sans confirmation.
- Pas de métriques inventées ou arrondies à la louche ; pas de conclusion sur un facteur sans l'expérience correspondante.
- Pas de réglage sur le test, pas de seuil choisi sur le test, pas d'augmentation sur val/test.

## Exigences de fond
- Étude **approfondie** des hyperparamètres et des variantes d'équations (pertes, optimiseurs, architectures) pour **CNN, CNN-LSTM, MLP** : c'est le cœur du projet.
- LR et RF restent dans la comparaison finale, comme références (réglage limité).
- Documenter aussi les expériences qui échouent ou n'aident pas.
- Hypothèse CNN-LSTM formulée comme testable : « le LSTM améliore-t-il la généralisation par rapport au CNN seul ? » — à vérifier expérimentalement, jamais affirmé.

## Exécution
- Jobs longs en terminal asynchrone, journaux dans `results/*.txt` ; reprise automatique (index d'expériences).
- CPU uniquement sur cette machine (16 threads logiques, 27,7 Go RAM, pas de CUDA).
- Environnement : `PROJET\.venv` (Python 3.14, torch CPU). Pas de PyYAML : configs en JSON.
