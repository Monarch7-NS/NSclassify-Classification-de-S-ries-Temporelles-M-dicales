import os
import wfdb
import numpy as np
from scipy.signal import butter, sosfiltfilt

# 1. Dossier où se trouvent les fichiers mitdb (ex: 'data/mitdb' ou 'data')
DATA_DIR = 'data'
OUTPUT_DIR = 'data'
WINDOW_SIZE = 180  # 180 avant + 180 après = 360 points (1 seconde à 360 Hz)
RECORD_ID = os.path.join(DATA_DIR, '200')

# 1. Lire en-tête et annotations
record = wfdb.rdrecord(RECORD_ID)
annotation = wfdb.rdann(RECORD_ID, 'atr')

# 2. Afficher les 10 premiers pics et lettres
print("--- 10 PREMIERS BATTEMENTS ---")
for i in range(10):
    sample = annotation.sample[i]
    symbol = annotation.symbol[i]
    time_sec = sample / record.fs
    print(f"Battement {i}: Index = {sample} | Temps = {time_sec:.2f}s | Lettre = '{symbol}'")

# 3. Extraire le battement à l'index X
idx_pic = int(annotation.sample[1])

if idx_pic >= WINDOW_SIZE and idx_pic + WINDOW_SIZE <= len(record.p_signal):
    beat_segment = record.p_signal[
        idx_pic - WINDOW_SIZE : idx_pic + WINDOW_SIZE, 0
    ]

    print("\n--- FORME DU SEGMENT EXTRAIT ---")
    print(f"Taille du segment : {beat_segment.shape} points")
    print(f"5 premières valeurs : {beat_segment[:5]}")
    print(f"Valeur au centre : {beat_segment[WINDOW_SIZE]:.3f}")
else:
    print("Impossible d'extraire une fenêtre complète autour de ce battement.")
print(f"Symbole du battement : {annotation.symbol[1]}")
print(f"Temps du battement : {idx_pic / record.fs:.2f} secondes")