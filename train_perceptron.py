"""
===============================================
  PERCEPTRON ALFABET TRAINER — Kaggle Dataset
  Compatible dengan detect_alfabet.py temen lo
===============================================
  Cara pakai:
    1. Taruh file CSV di folder yang sama:
       A_Z Handwritten Data.csv
    2. Jalankan: python train_perceptron.py

  CSV otomatis dicari di:
    - folder yang sama
    - subfolder dataset/
    - subfolder data/
===============================================
"""

import numpy as np
import pickle
import os
import sys
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

IMG_SIZE   = 28
LABELS     = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MODEL_PATH  = "perceptron_alfabet.pkl"
SCALER_PATH = "scaler_alfabet.pkl"
VOKAL       = set("AEIOU")

# Berapa sampel per huruf yang dipakai untuk training
# Naikin angka ini = lebih akurat tapi lebih lama
# Rekomendasikan: 1500 (balance antara kecepatan & akurasi)
SAMPLES_PER_CLASS = 1500


# ─── CARI FILE CSV ─────────────────────────────────────────
def find_csv():
    candidates = [
        "A_Z Handwritten Data.csv",
        "dataset/A_Z Handwritten Data.csv",
        "data/A_Z Handwritten Data.csv",
        "A_Z_Handwritten_Data.csv",
        "dataset/A_Z_Handwritten_Data.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    # Cari file csv apapun yang ada di folder saat ini
    for f in os.listdir("."):
        if f.endswith(".csv"):
            print(f"   ⚠️  Ketemu CSV: '{f}' — akan dicoba dipakai")
            return f
    for folder in ["dataset", "data"]:
        if os.path.isdir(folder):
            for f in os.listdir(folder):
                if f.endswith(".csv"):
                    path = os.path.join(folder, f)
                    print(f"   ⚠️  Ketemu CSV: '{path}' — akan dicoba dipakai")
                    return path
    return None


# ─── FITUR HOG (identik dgn detect_alfabet.py temen lo) ───
def extract_hog_features(img_flat):
    img      = img_flat.reshape(IMG_SIZE, IMG_SIZE)
    img_u8   = (img * 255).astype(np.uint8)
    winSize     = (IMG_SIZE, IMG_SIZE)
    blockSize   = (IMG_SIZE//2, IMG_SIZE//2)
    blockStride = (IMG_SIZE//4, IMG_SIZE//4)
    cellSize    = (IMG_SIZE//4, IMG_SIZE//4)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    return hog.compute(img_u8).flatten()

def extract_features(img_flat):
    """Pixel (784) + HOG (324) = 1108 fitur — sama persis kayak detect_alfabet.py"""
    return np.concatenate([img_flat, extract_hog_features(img_flat)])


# ─── LOAD DATASET KAGGLE ───────────────────────────────────
def load_kaggle_dataset(csv_path, samples_per_class=SAMPLES_PER_CLASS):
    """
    Format CSV Kaggle: kolom pertama = label (0-25), sisanya = 784 pixel (28x28)
    """
    print(f"\n📂 Membaca dataset: {csv_path}")
    print(f"   Sampel per huruf : {samples_per_class}")
    print(f"   Total target     : {samples_per_class * 26:,} baris")
    print(f"   (membaca CSV per chunk, sabar ya...)\n")

    import pandas as pd

    X_all = {i: [] for i in range(26)}
    chunk_size = 50_000
    total_read = 0

    for chunk in pd.read_csv(csv_path, header=None, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            label = int(row.iloc[0])
            if len(X_all[label]) < samples_per_class:
                pixels = row.iloc[1:].values.astype(np.float32) / 255.0
                X_all[label].append(pixels)

        total_read += len(chunk)
        filled = sum(1 for v in X_all.values() if len(v) >= samples_per_class)
        print(f"   Dibaca: {total_read:,} baris | Huruf penuh: {filled}/26", end='\r')

        # Berhenti kalau semua kelas sudah cukup
        if all(len(v) >= samples_per_class for v in X_all.values()):
            break

    print()

    # Gabungkan
    X, y = [], []
    for label_idx in range(26):
        samples = X_all[label_idx]
        if len(samples) == 0:
            print(f"   ⚠️  Huruf {LABELS[label_idx]} tidak ada datanya!")
            continue
        if len(samples) < samples_per_class:
            print(f"   ⚠️  Huruf {LABELS[label_idx]}: hanya {len(samples)} sampel (kurang dari {samples_per_class})")
        X.extend(samples)
        y.extend([label_idx] * len(samples))

    print(f"\n✅ Dataset dimuat: {len(X):,} sampel total")
    return np.array(X), np.array(y)


# ─── MAIN TRAIN ────────────────────────────────────────────
def train_model():
    print()
    print("╔══════════════════════════════════════════╗")
    print("║  🧠 PERCEPTRON ALFABET TRAINER           ║")
    print("║  Dataset: Kaggle A-Z Handwritten         ║")
    print("╚══════════════════════════════════════════╝")

    # Cari CSV
    csv_path = find_csv()
    if csv_path is None:
        print("\n❌ File CSV tidak ditemukan!")
        print("   Taruh 'A_Z Handwritten Data.csv' di folder yang sama")
        print("   atau di subfolder 'dataset/' atau 'data/'")
        sys.exit(1)

    # Load data
    X_raw, y = load_kaggle_dataset(csv_path)

    # Ekstraksi fitur (pixel + HOG)
    print(f"\n🔄 Ekstraksi fitur pixel + HOG...")
    print(f"   Ini butuh beberapa menit untuk {len(X_raw):,} sampel...")
    X = []
    for i, x in enumerate(X_raw):
        X.append(extract_features(x))
        if (i+1) % 1000 == 0:
            print(f"   Progress: {i+1:,}/{len(X_raw):,}", end='\r')
    X = np.array(X)
    print(f"\n   ✅ Dimensi fitur: {X.shape[1]} (784 pixel + 324 HOG = 1108)")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"\n   Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Scaler
    print("\n🔄 Normalisasi StandardScaler...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Train MLP (arsitektur sama persis kayak temen lo)
    print("\n🔄 Training MLP Perceptron...")
    print("   Arsitektur: Input(1108) → 512 → 256 → 128 → Output(26)")
    print("   Optimizer : Adam | Aktivasi: ReLU | Early Stopping: ON\n")

    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=300,
        random_state=42,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    model.fit(X_train_s, y_train)

    # Evaluasi
    y_pred   = model.predict(X_test_s)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  ✅ AKURASI MODEL: {accuracy*100:.2f}%")
    print(f"{'='*50}")

    print("\n📊 Akurasi per huruf:")
    report = classification_report(y_test, y_pred, target_names=LABELS, output_dict=True)
    for letter in LABELS:
        acc   = report[letter]['precision']
        jenis = "🔴 VOKAL" if letter in VOKAL else "🔵 KONS "
        bar   = "█" * int(acc * 20)
        print(f"   {jenis} {letter}: {bar:<20} {acc*100:.1f}%")

    # Simpan
    print(f"\n💾 Menyimpan model ke '{MODEL_PATH}' dan '{SCALER_PATH}'...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    print("✅ Model berhasil disimpan!")
    print(f"\n▶  Sekarang jalankan: python detect_alfabet.py")
    print(f"   (atau python run.py)\n")


if __name__ == "__main__":
    train_model()
