import os
import wfdb
import numpy as np
from scipy.signal import butter, sosfiltfilt
import warnings
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit

def filter_ecg(signal, fs=360):
    sos = butter(
        N=4,
        Wn=[0.5, 40],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    return sosfiltfilt(sos, signal)

# 1. Dossier où se trouvent les fichiers mitdb (ex: 'data/mitdb' ou 'data')
DATA_DIR = os.path.join('data', 'mitdb')
OUTPUT_DIR = 'data'
WINDOW_SIZE = 180  # 180 avant + 180 après = 360 points (1 seconde à 360 Hz)

# 2. Répartition stricte par patient (Protocole De Chazal)
BEAT_SYMBOLS = {
    'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S',
    'V', 'r', 'F', 'e', 'j', 'n', 'E', '/',
    'f', 'Q'
}
DS1_TRAIN = [
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
    '122', '124', '201','203', '205', '207', '208', '209', '215', '220',
    '223', '230'
]

DS2_TEST = [
    '100', '103', '105', '111', '113', '117', '121', '123', '200','202',
    '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
    '233', '234'
]

# 3. Mapping binaire : 0 = Normal, 1 = Anormal
BINARY_MAPPING = {
    # Normal (0)
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    
    # Anormal (1)
    'A': 1, 'a': 1, 'J': 1, 'S': 1,  # Supraventriculaire
    'V': 1, 'E': 1,                  # Ventriculaire
    'F': 1,                          # Fusion
    '/': 1, 'f': 1, 'Q': 1           # Inconnu / Rythme entraîné (paced)
}

def extract_dataset(record_list, data_dir, window_size=180):
    X = []
    y = []
    RR_features = []
    symbols = [] #symbols garde le type d’arythmie d’origine (N, V, A...) pour l’analyse des confusions.
    patient_ids = []  #patient_ids permet le split de validation par patient et l’analyse par patient.


    for rec_id in record_list:
        rec_path = os.path.join(data_dir, rec_id)

        if not os.path.exists(rec_path + '.hea'):
            warnings.warn(f"Fichier introuvable pour l'enregistrement {rec_id} : {rec_path}.hea")
            continue

        # Lecture du signal et des annotations
        record = wfdb.rdrecord(rec_path)
        annotation = wfdb.rdann(rec_path, 'atr')

        if 'MLII' not in record.sig_name: #dans l’enregistrement 114, l’ordre des dérivations est inversé (V5 en colonne 0, MLII en colonne 1). Avec [:, 0] tu aurais mélangé deux dérivations différentes dans ton dataset si on fait pas le if ici
            warnings.warn(f"Enregistrement {rec_id} : pas de dérivation MLII ({record.sig_name}), ignoré")
            continue
        lead_idx = record.sig_name.index('MLII')
        signal = record.p_signal[:, lead_idx]

        # Filtrage du signal complet
        signal = filter_ecg(signal, fs=record.fs)

        # Normalisation Z-score
        signal = (
            (signal - np.mean(signal))
            / (np.std(signal) + 1e-8)
        )

        total_samples = len(signal)

        # Positions des annotations appartenant au mapping binaire
        # 1. Récupérer les annotations correspondant aux battements
        all_beats = [
            (int(sample), symbol)
            for sample, symbol in zip(annotation.sample, annotation.symbol)
            if symbol in BEAT_SYMBOLS
        ]

        # 2. Calculer les intervalles RR sur tous ces battements
        all_samples = np.array(
            [sample for sample, symbol in all_beats],
            dtype=np.int64
        )

        all_rr = np.diff(all_samples) / record.fs

        # 3. Garder seulement les battements pour lesquels on connaît la classe binaire
        valid_beats = [
            (sample, symbol, i)
            for i, (sample, symbol) in enumerate(all_beats)
            if symbol in BINARY_MAPPING
        ]

        
        for r_peak, symbol, beat_idx in valid_beats:
            if beat_idx == 0 or beat_idx >= len(all_beats) - 1:
                continue


            start = r_peak - window_size
            end = r_peak + window_size

            # Vérifier les limites du signal
            if start < 0 or end > total_samples:
                continue

            beat = signal[start:end]

            if len(beat) != 2 * window_size:
                continue

            label = BINARY_MAPPING[symbol]

            # Intervalle entre le battement précédent et le battement courant
            rr_pre = all_rr[beat_idx - 1]

            # Intervalle entre le battement courant et le battement suivant
            rr_post = all_rr[beat_idx]

            # Moyenne des intervalles précédant rr_pre
            history_start = max(0, beat_idx - 6)
            previous_rr = all_rr[history_start:beat_idx - 1]

            if len(previous_rr) > 0:
                rr_reference = np.mean(previous_rr)
            else:
                rr_reference = rr_pre

            rr_ratio = rr_pre / (rr_reference + 1e-8)

            X.append(beat)
            y.append(label)
            RR_features.append([rr_pre, rr_post, rr_ratio])
            '''une valeur est ajoutée uniquement pour les battements réellement retenus, 
            donc les 5 listes (X, y, RR_features, symbols, patient_ids) restent alignées.'''
            symbols.append(symbol) 
            patient_ids.append(rec_id) 

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int64),
        np.array(RR_features, dtype=np.float32),
        np.array(symbols, dtype='<U1'),
        np.array(patient_ids, dtype='<U3')
    )



if __name__ == '__main__':

    print("Extraction du jeu d'entraînement (DS1)...")
    X_train, y_train, RR_train, sym_train, pid_train = extract_dataset(
    DS1_TRAIN, DATA_DIR, WINDOW_SIZE
    )

    print("Extraction du jeu de test (DS2)...")
    X_test, y_test, RR_test, sym_test, pid_test = extract_dataset(
    DS2_TEST, DATA_DIR, WINDOW_SIZE
    )

    # Découpage train / validation PAR PATIENT à l'intérieur de DS1
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, val_idx = next(gss.split(X_train, y_train, groups=pid_train))

    # D'abord on extrait la validation à partir de l'ensemble complet DS1
    X_val, y_val, RR_val = X_train[val_idx], y_train[val_idx], RR_train[val_idx]
    sym_val, pid_val = sym_train[val_idx], pid_train[val_idx]

    # Ensuite on réduit le train aux indices restants
    X_train, y_train, RR_train = X_train[tr_idx], y_train[tr_idx], RR_train[tr_idx]
    sym_train, pid_train = sym_train[tr_idx], pid_train[tr_idx]

    # Vérification : aucun patient en commun
    assert set(pid_train).isdisjoint(set(pid_val)), "Fuite : patient présent dans train et val !"
    print("Patients train :", sorted(set(pid_train)))
    print("Patients val   :", sorted(set(pid_val)))

    print("\n========== STATISTIQUES ==========")
    print(f"X_train : {X_train.shape}")
    print(f"y_train : {y_train.shape}")
    print(f"RR_train: {RR_train.shape}")

    print(f"X_test  : {X_test.shape}")
    print(f"y_test  : {y_test.shape}")
    print(f"RR_test : {RR_test.shape}")

    print(f"X_val   : {X_val.shape}")
    print(f"y_val   : {y_val.shape}")
    print(f"RR_val  : {RR_val.shape}")


    print(
        f"\nTrain - Normaux (0): {np.sum(y_train == 0)}, "
        f"Anormaux (1): {np.sum(y_train == 1)}"
    )
    print(
        f"Test - Normaux (0): {np.sum(y_test == 0)}, "
        f"Anormaux (1): {np.sum(y_test == 1)}"
    )

    print(
        f"Val - Normaux (0): {np.sum(y_val == 0)}, "
        f"Anormaux (1): {np.sum(y_val == 1)}"
    )

    print("\nSymboles train :", dict(Counter(sym_train)))
    print("Symboles test  :", dict(Counter(sym_test)))
    print("Battements par patient (train) :", dict(Counter(pid_train)))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'mitdb_train.npz'),
        X=X_train, y=y_train, RR=RR_train,
        symbol=sym_train, patient_id=pid_train
    )

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'mitdb_val.npz'),
        X=X_val, y=y_val, RR=RR_val,
        symbol=sym_val, patient_id=pid_val
    )

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'mitdb_test.npz'),
        X=X_test, y=y_test, RR=RR_test,
        symbol=sym_test, patient_id=pid_test
    )

    print("\nDatasets sauvegardés avec les caractéristiques RR !")
