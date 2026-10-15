import copy
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

BATCH_SIZE = 128
MAX_EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
CLIP_GRAD = 1.0
SEEDS = [42, 43, 44]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device utilisé : {DEVICE}")


def set_seed(seed):
    """Fixe les graines pour rendre l'expérience reproductible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproductibilité accrue, au prix de performances parfois moindres.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. DATASET
# ============================================================

class ECGDataset(Dataset):
    def __init__(self, npz_path):
        with np.load(npz_path) as data:
            X = np.asarray(data["X"], dtype=np.float32)
            RR = np.asarray(data["RR"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=np.float32)

        # X doit avoir la forme (N, longueur) ou (N, longueur, 1).
        if X.ndim == 2:
            X = X[..., np.newaxis]
        elif X.ndim != 3 or X.shape[-1] != 1:
            raise ValueError(
                f"Forme X invalide dans {npz_path} : {X.shape}"
            )

        # RR doit avoir la forme (N, 3).
        if RR.ndim == 1:
            RR = RR.reshape(-1, 1)

        if RR.ndim != 2 or RR.shape[1] != 3:
            raise ValueError(
                f"RR doit avoir 3 caractéristiques, forme reçue : "
                f"{RR.shape}"
            )

        # Labels binaires sous forme (N, 1).
        y = y.reshape(-1, 1)

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("Les labels y doivent être binaires (0 ou 1).")

        if not (len(X) == len(RR) == len(y)):
            raise ValueError("X, RR et y ont des tailles différentes.")

        if not (
            np.isfinite(X).all()
            and np.isfinite(RR).all()
            and np.isfinite(y).all()
        ):
            raise ValueError("Des valeurs NaN ou infinies sont présentes.")

        self.X = torch.from_numpy(X)
        self.RR = torch.from_numpy(RR)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.RR[idx], self.y[idx]


train_dataset = ECGDataset("data/mitdb_train.npz")
val_dataset = ECGDataset("data/mitdb_val.npz")
test_dataset = ECGDataset("data/mitdb_test.npz")

print("Taille train :", len(train_dataset))
print("Taille validation :", len(val_dataset))
print("Taille test :", len(test_dataset))

# Vérifier que chaque ensemble possède bien les deux classes.
y_train = train_dataset.y.numpy().ravel()
y_val = val_dataset.y.numpy().ravel()
y_test = test_dataset.y.numpy().ravel()

if len(np.unique(y_train)) != 2:
    raise ValueError("Le train doit contenir les deux classes.")

if len(np.unique(y_val)) != 2:
    raise ValueError("La validation doit contenir les deux classes.")

print("Répartition train :", np.bincount(y_train.astype(int)))
print("Répartition validation :", np.bincount(y_val.astype(int)))
print("Répartition test :", np.bincount(y_test.astype(int)))


# ============================================================
# 3. MODELE LSTM
# ============================================================

class PureLSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size + 3, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x, rr):
        # x : (batch, longueur, 1)
        # rr : (batch, 3)
        _, (hn, _) = self.lstm(x)

        # Dernière couche LSTM : (batch, hidden_size)
        lstm_out = hn[-1]

        combined = torch.cat((lstm_out, rr), dim=1)
        return self.fc(combined)


# ============================================================
# 4. FONCTIONS D'EVALUATION
# ============================================================

def predict_probabilities(model, loader):
    """Retourne les probabilités de la classe positive et les labels."""
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for X_batch, RR_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            RR_batch = RR_batch.to(DEVICE)

            logits = model(X_batch, RR_batch)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy().ravel())
            all_targets.append(y_batch.numpy().ravel())

    return (
        np.concatenate(all_probs),
        np.concatenate(all_targets).astype(int),
    )


def find_best_f1_threshold(y_true, probabilities):
    """Choisit le seuil qui maximise le F1 sur la validation."""
    # Inclut tous les seuils distincts possibles, ainsi que les bornes.
    thresholds = np.unique(
        np.concatenate(([0.0, 1.0], probabilities))
    )

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        preds = (probabilities >= threshold).astype(int)

        score = f1_score(
            y_true, preds, zero_division=0
        )

        # En cas d'égalité, conserver le premier seuil optimal.
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def compute_metrics(y_true, probabilities, threshold):
    preds = (probabilities >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(
            y_true, preds, zero_division=0
        ),
        "recall": recall_score(
            y_true, preds, zero_division=0
        ),
        "f1": f1_score(y_true, preds, zero_division=0),
        "pr_auc": average_precision_score(
            y_true, probabilities
        ),
        "confusion_matrix": confusion_matrix(
            y_true, preds, labels=[0, 1]
        ),
    }


# ============================================================
# 5. ENTRAINEMENT + EARLY STOPPING SUR LA PR-AUC
# ============================================================

def train_one_seed(seed):
    set_seed(seed)

    # Générateur dédié pour le mélange reproductible des données.
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = PureLSTMClassifier().to(DEVICE)

    # BCE pondérée : poids positif = N_neg / N_pos.
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))

    pos_weight = torch.tensor(
        [n_neg / n_pos],
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_pr_auc = -float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"\n{'=' * 60}")
    print(f"ENTRAINEMENT - SEED {seed}")
    print(f"pos_weight = {n_neg}/{n_pos} = {n_neg / n_pos:.4f}")
    print(f"{'=' * 60}")

    for epoch in range(MAX_EPOCHS):
        # ---------------- ENTRAINEMENT ----------------
        model.train()
        total_train_loss = 0.0

        for X_batch, RR_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            RR_batch = RR_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(X_batch, RR_batch)
            loss = criterion(logits, y_batch)

            loss.backward()

            # Gradient clipping obligatoire pour le LSTM.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=CLIP_GRAD
            )

            optimizer.step()

            total_train_loss += loss.item() * X_batch.size(0)

        train_loss = total_train_loss / len(train_dataset)

        # ---------------- VALIDATION ----------------
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for X_batch, RR_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                RR_batch = RR_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                logits = model(X_batch, RR_batch)
                loss = criterion(logits, y_batch)

                total_val_loss += loss.item() * X_batch.size(0)

        val_loss = total_val_loss / len(val_dataset)

        # La sélection du modèle dépend de la PR-AUC de validation.
        val_probs, val_targets = predict_probabilities(
            model, val_loader
        )
        val_pr_auc = average_precision_score(
            val_targets, val_probs
        )

        print(
            f"Époque [{epoch + 1:02d}/{MAX_EPOCHS}] "
            f"- Train Loss: {train_loss:.4f} "
            f"- Val Loss: {val_loss:.4f} "
            f"- Val PR-AUC: {val_pr_auc:.4f}"
        )

        # Amélioration stricte de la PR-AUC.
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch + 1

            # Copie indépendante des paramètres du meilleur modèle.
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # ---------------- EARLY STOPPING ----------------
        if epochs_without_improvement >= PATIENCE:
            print(
                f"Early stopping à l'époque {epoch + 1}. "
                f"Meilleure époque : {best_epoch}."
            )
            break

    # Restaurer le meilleur modèle de validation.
    model.load_state_dict(best_state)

    # Sauvegarde du meilleur modèle pour cette seed.
    checkpoint_path = f"best_lstm_seed_{seed}.pth"
    torch.save(model.state_dict(), checkpoint_path)

    # Déterminer le seuil uniquement sur la validation.
    val_probs, val_targets = predict_probabilities(
        model, val_loader
    )
    threshold, val_best_f1 = find_best_f1_threshold(
        val_targets, val_probs
    )

    # Le test n'est consulté qu'après la sélection du modèle
    # et la détermination du seuil.
    test_probs, test_targets = predict_probabilities(
        model, test_loader
    )

    test_metrics = compute_metrics(
        test_targets, test_probs, threshold
    )

    print(f"\n--- SEED {seed} : RESULTATS ---")
    print(f"Meilleure époque       : {best_epoch}")
    print(f"Meilleure Val PR-AUC   : {best_val_pr_auc:.4f}")
    print(f"Meilleur F1 validation : {val_best_f1:.4f}")
    print(f"Seuil choisi           : {threshold:.6f}")
    print(f"Test PR-AUC            : {test_metrics['pr_auc']:.4f}")
    print(f"Test Accuracy          : {test_metrics['accuracy']:.4f}")
    print(f"Test Precision         : {test_metrics['precision']:.4f}")
    print(f"Test Recall             : {test_metrics['recall']:.4f}")
    print(f"Test F1                 : {test_metrics['f1']:.4f}")
    print("Matrice de confusion :")
    print(test_metrics["confusion_matrix"])

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "val_pr_auc": best_val_pr_auc,
        "val_best_f1": val_best_f1,
        "threshold": threshold,
        **{
            key: value
            for key, value in test_metrics.items()
            if key != "confusion_matrix"
        },
        "confusion_matrix": test_metrics["confusion_matrix"],
    }


# ============================================================
# 6. EVALUATION FINALE MULTI-SEED
# ============================================================

results = []

for seed in SEEDS:
    results.append(train_one_seed(seed))

metric_names = [
    "pr_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
]

print("\n" + "=" * 60)
print("SYNTHESE FINALE SUR LE TEST - 3 SEEDS")
print("=" * 60)

for name in metric_names:
    values = np.array([result[name] for result in results])

    print(
        f"{name:10s} : "
        f"{values.mean():.4f} ± {values.std(ddof=1):.4f} "
        f"(moyenne ± écart-type)"
    )

print("\nDétail des seuils choisis sur validation :")
for result in results:
    print(
        f"Seed {result['seed']} : "
        f"seuil={result['threshold']:.6f}, "
        f"époque={result['best_epoch']}, "
        f"Val PR-AUC={result['val_pr_auc']:.4f}"
    )
