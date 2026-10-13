# NSclassify — Classification de séries temporelles médicales (ECG)

Classification multi-classes de battements ECG (MIT-BIH Arrhythmia Database) en 5 classes d'arythmies AAMI, avec deux exigences : **performance** (contexte médical, classes minoritaires critiques) et **interprétabilité** (Grad-CAM temporel, poids d'attention, filtres).

## Démarrage rapide

```bash
bash setup_env.sh                       # venv + dépendances + téléchargement MIT-BIH
source .venv/bin/activate               # Windows : .venv\Scripts\activate

python -m src.baseline                  # baseline : features RR/QRS + Random Forest
python -m src.train --model cnn --augment
python -m src.train --model resnet --augment
python -m src.train --model cnn_lstm --augment
python -m src.evaluate results/resnet_aug/best.pt
pytest                                  # tests de fumée (données synthétiques)
```

Sans le script : `pip install -r requirements.txt && python scripts/download_data.py` (option `--ptbdb` pour PTB-DB).

## Structure

```
README.md
requirements.txt
setup_env.sh
scripts/download_data.py      téléchargement PhysioNet (wfdb)
notebooks/demo.ipynb          EDA, augmentation, prédiction + explication
src/
  data_processing.py          filtrage, segmentation, étiquettes AAMI, split par patient, tableau descriptif
  features.py                 features hand-crafted (intervalles RR, morphologie QRS)
  augmentation.py             jittering, time warping, magnitude warping
  models.py                   CNN1D, ResNet1D, LSTM, CNN-LSTM + attention
  train.py                    entraînement (pondération/échantillonnage, early stopping, seeds)
  evaluate.py                 sensibilité, spécificité, VPP, VPN, F1, confusion, robustesse au bruit
  interpretability.py         Grad-CAM 1D, attention, filtres, activations
  baseline.py                 Random Forest sur features hand-crafted
tests/test_smoke.py
data/                         raw/ (PhysioNet) et processed/ (cache .npz) — non versionné
results/                      checkpoints, métriques, figures — non versionné
```

## Données et prétraitement

| Élément | Valeur |
|---|---|
| Source | MIT-BIH Arrhythmia Database, 48 enregistrements de 30 min |
| Fréquence d'échantillonnage | 360 Hz |
| Dérivation | MLII (repérée par son nom, y compris l'enr. 114 où les voies sont inversées) |
| Filtrage | Butterworth passe-bande 0.5–40 Hz (zéro-phase) |
| Segment | 360 échantillons (1 s), centré sur le pic R annoté |
| Normalisation | z-score par battement |
| Exclusions | enregistrements à rythme stimulé 102, 104, 107, 217 (convention AAMI) |

Classes AAMI : **N** (normal, blocs de branche, échappements), **S** (ectopique supraventriculaire), **V** (ectopique ventriculaire), **F** (fusion), **Q** (inconnu/stimulé). Le tableau exact (effectifs par classe et par split, déséquilibre) est produit par `describe_dataset`, affiché au début de chaque entraînement et dans le notebook. La classe N domine très fortement ; F et S sont minoritaires.

## Reproductibilité et protocole

- **Split par patient, jamais par battement** (de Chazal et al., 2004) : test = DS2 (22 patients) ; train/val = patients de DS1 (validation = 20 % des patients de DS1 tirés avec la seed). Aucun patient n'apparaît dans deux splits.
- Seeds fixées (`--seed`, défaut 42) pour Python, NumPy, PyTorch, DataLoader.
- Sélection du modèle et early stopping sur le **F1 macro de validation** (l'accuracy est trompeuse avec ce déséquilibre).
- Déséquilibre : pondération de la loss (inverse de fréquence amorti, `--balance weights`), échantillonnage pondéré (`sampler`) ou rien (`none`). Augmentation (`--augment`) : time warping, magnitude warping, jittering, puis re-normalisation.

## Modèles

| Nom | Description |
|---|---|
| Random Forest | 13 features (RR précédent/suivant/local/ratio, amplitude R, largeur et aire du QRS, pente max, profondeurs Q/S, écart-type, skewness, kurtosis), `class_weight=balanced_subsample` |
| `cnn` | 1D-CNN à 3 blocs, pooling global |
| `resnet` | ResNet1D, blocs résiduels (Wang et al., 2017) |
| `lstm` | BiLSTM sur le signal sous-échantillonné |
| `cnn_lstm` | CNN → BiLSTM → attention additive |

## Évaluation

Par classe (un-contre-tous) : sensibilité, spécificité, VPP, VPN, F1, plus matrice de confusion normalisée et robustesse au bruit blanc (SNR 30 → 0 dB). Sorties dans `results/<tag>/` : `metrics_per_class.csv`, `confusion_matrix.png`, `noise_robustness.csv`, `test_summary.json`, `history.json`, `best.pt`.

## Interprétabilité

- **Grad-CAM temporel** (`GradCAM1D`) : CNN, ResNet1D et CNN-LSTM.
- **Poids d'attention** du CNN-LSTM (`attention_weights`, `plot_attention`).
- **Filtres appris** de la première couche et activations intermédiaires.

## Pistes non implémentées

Transfer learning depuis ImageNet sur spectrogrammes, PTB-DB (le téléchargement est prévu avec `--ptbdb`, pas le pipeline de segmentation), recherche d'hyperparamètres.

## Références

- Fawaz et al. (2019), *Deep learning for time series classification: a review* — https://arxiv.org/abs/1809.04356
- Wang et al. (2017), *Time series classification from scratch with deep neural networks: a strong baseline* — https://arxiv.org/abs/1611.06455
- Moody & Mark (2001), MIT-BIH Arrhythmia Database — https://physionet.org/content/mitdb/1.0.0/
- de Chazal et al. (2004), *Automatic classification of heartbeats using ECG morphology and heartbeat interval features*, IEEE TBME.


## PARTIE DATA - MARIEM 
Sur la branche dev:
- ** essai.py — Exploration et test d'extraction

Ce script sert à vérifier que les fichiers MIT-BIH sont correctement lus avec wfdb. Il affiche les premières annotations ECG, leur position et leur temps, puis extrait une fenêtre d'une seconde autour d'une annotation pour vérifier sa taille et ses valeurs.



- ** preprocess.py — Prétraitement et préparation du dataset

Ce script prépare les données pour le machine learning : il filtre et normalise les signaux ECG, extrait des fenêtres autour des battements, calcule des caractéristiques temporelles RR, attribue les étiquettes binaires normal/anormal, sépare les données en entraînement, validation et test, puis sauvegarde les datasets au format .npz.
