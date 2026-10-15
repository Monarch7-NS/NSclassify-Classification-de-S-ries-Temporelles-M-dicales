# Instructions pour Galet (Copilot) — projet DATA931 / MIT-BIH

Ce fichier est chargé automatiquement. Avant toute action, lire `.github/knowledge/` :

1. `USER_PREFERENCES.md` — comment NS veut que Galet travaille (style, règles de scope, interdits).
2. `EXPERIMENTAL_RULES.md` — règles du protocole expérimental (non négociables).
3. `PROJECT_CONTEXT.md` — contexte, jeu de données, structure du dépôt, commandes.
4. `MODEL_STUDY.md` — équations et hypothèses testables pour MLP / CNN / CNN-LSTM / pertes / optimiseurs.
5. `DECISIONS.md` — décisions prises (avec raisons) ; à mettre à jour à chaque nouvelle décision.

Règles résumées :
- Aucune métrique inventée : tout chiffre cité vient de `results/experiments/index.csv` ou d'un `metrics.json`.
- Jamais d'entraînement avant que split, étiquettes, prétraitement et tests anti-fuite passent (`python -m pytest tests -q`).
- Le test n'est évalué qu'à l'étape `final`, une fois, avec seuil figé sur la validation.
- Aucun `git commit` / `git push` sans confirmation explicite de NS.
- Réponses courtes, impersonnelles, sans emoji.
- Mettre à jour `.github/knowledge/` à chaque décision, résultat ou préférence nouvelle.
