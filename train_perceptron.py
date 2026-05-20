"""
===============================================
  PERCEPTRON ALFABET TRAINER (Fixed Windows)
===============================================
"""

import numpy as np
import pickle
import os
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image, ImageDraw, ImageFont
import cv2

IMG_SIZE         = 28
SAMPLES_PER_CLASS = 200
LABELS           = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MODEL_PATH       = "perceptron_alfabet.pkl"
SCALER_PATH      = "scaler_alfabet.pkl"
VOKAL            = set("AEIOU")

# ─── CARI FONT YANG ADA ────────────────────────────────────
def get_font(size):
    """Cari font yang tersedia di sistem (Windows/Linux/Mac)."""
    candidates = [
        # Windows
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/verdanab.ttf",
        "C:/Windows/Fonts/tahomabd.ttf",
        "C:/Windows/Fonts/consolab.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
    # Fallback: default font (kecil tapi pasti ada)
    return ImageFont.load_default()

# ─── GENERATE GAMBAR HURUF ─────────────────────────────────
def generate_letter_image(letter, size=IMG_SIZE, noise_level=0.15):
    scale = 4
    big   = size * scale
    
    pil_img = Image.new('L', (big, big), color=0)
    draw    = ImageDraw.Draw(pil_img)
    
    font_size = int(big * np.random.uniform(0.55, 0.80))
    font = get_font(font_size)
    
    brightness = np.random.randint(200, 255)
    
    bbox = draw.textbbox((0, 0), letter, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    ox = np.random.randint(-big//8, big//8)
    oy = np.random.randint(-big//8, big//8)
    x  = (big - w) // 2 + ox
    y  = (big - h) // 2 + oy
    
    draw.text((x, y), letter, fill=brightness, font=font)
    
    pil_img = pil_img.resize((size, size), Image.LANCZOS)
    img     = np.array(pil_img, dtype=np.float32) / 255.0
    
    # Noise
    noise = np.random.normal(0, noise_level, img.shape)
    img   = np.clip(img + noise, 0, 1)
    
    # Rotasi kecil
    angle = np.random.uniform(-12, 12)
    M     = cv2.getRotationMatrix2D((size//2, size//2), angle, 1.0)
    img   = cv2.warpAffine(img, M, (size, size))
    
    return img.flatten()

# ─── FITUR HOG ─────────────────────────────────────────────
def extract_hog_features(img_flat, size=IMG_SIZE):
    img      = img_flat.reshape(size, size)
    img_u8   = (img * 255).astype(np.uint8)
    hog = cv2.HOGDescriptor(
        (size, size), (size//2, size//2),
        (size//4, size//4), (size//4, size//4), 9
    )
    return hog.compute(img_u8).flatten()

def extract_features(img_flat):
    return np.concatenate([img_flat, extract_hog_features(img_flat)])

# ─── GENERATE DATASET ──────────────────────────────────────
def generate_dataset():
    print("🔄 Membuat dataset sintetis huruf A-Z...")
    
    # Cek font dulu
    test_font = get_font(40)
    if hasattr(test_font, 'path'):
        print(f"   Font: {test_font.path}")
    else:
        print("   Font: default (PIL built-in)")
    
    X, y = [], []
    for i, letter in enumerate(LABELS):
        print(f"   [{i+1:2d}/26] Generating '{letter}'...", end='\r')
        for _ in range(SAMPLES_PER_CLASS):
            feat = generate_letter_image(letter, noise_level=np.random.uniform(0.03, 0.20))
            X.append(feat)
            y.append(i)
    print("\n✅ Dataset selesai!")
    return np.array(X), np.array(y)

# ─── TRAIN ─────────────────────────────────────────────────
def train_model():
    print("\n" + "="*50)
    print("  🧠 PERCEPTRON ALFABET TRAINER")
    print("="*50)

    X_raw, y = generate_dataset()

    print("\n🔄 Ekstraksi fitur (pixel + HOG)...")
    X = np.array([extract_features(x) for x in X_raw])
    print(f"   Dimensi fitur: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    print("\n🔄 Normalisasi...")
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print("\n🔄 Training MLP Perceptron...")
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
    model.fit(X_train_scaled, y_train)

    y_pred   = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  ✅ AKURASI MODEL: {accuracy*100:.2f}%")
    print(f"{'='*50}")

    print("\n📊 Akurasi per huruf:")
    report = classification_report(y_test, y_pred, target_names=LABELS, output_dict=True)
    for letter in LABELS:
        acc  = report[letter]['precision']
        jenis = "🔴 VOKAL" if letter in VOKAL else "🔵 KONS "
        bar  = "█" * int(acc * 20)
        print(f"   {jenis} {letter}: {bar:<20} {acc*100:.1f}%")

    print(f"\n💾 Menyimpan model...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    print("✅ Model berhasil disimpan!")
    print("\n▶  Jalankan: python detect_alfabet.py")
    return model, scaler

if __name__ == "__main__":
    train_model()
