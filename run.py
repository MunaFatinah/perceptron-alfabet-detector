"""
╔══════════════════════════════════════════════════════╗
║     PERCEPTRON ALFABET DETECTOR — RUNNER             ║
║     Jalankan file ini untuk start otomatis!          ║
╚══════════════════════════════════════════════════════╝
"""

import os
import sys
import subprocess

MODEL_PATH = "perceptron_alfabet.pkl"

def check_requirements():
    """Cek package yang dibutuhkan."""
    required = {
        'cv2':      'opencv-python',
        'numpy':    'numpy',
        'sklearn':  'scikit-learn',
        'PIL':      'Pillow',
    }
    missing = []
    for module, pkg in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)
    return missing


def main():
    print()
    print("╔══════════════════════════════════════════╗")
    print("║   🔤  PERCEPTRON ALFABET DETECTOR        ║")
    print("║   Deteksi A-Z | Vokal & Konsonan         ║")
    print("╠══════════════════════════════════════════╣")
    print("║  Step 1: Cek requirements                ║")
    print("║  Step 2: Train model (jika belum ada)    ║")
    print("║  Step 3: Buka kamera & deteksi huruf     ║")
    print("╚══════════════════════════════════════════╝")
    print()

    # 1. Cek requirements
    missing = check_requirements()
    if missing:
        print(f"❌ Package belum terinstall: {', '.join(missing)}")
        print("   Jalankan: pip install " + " ".join(missing))
        sys.exit(1)
    print("✅ Semua requirements sudah ada!")

    # 2. Train jika model belum ada
    if not os.path.exists(MODEL_PATH):
        print()
        print("📦 Model belum ada. Mulai training Perceptron...")
        print("   (Proses ini ~2-5 menit, cukup sekali saja)")
        print()
        result = subprocess.run([sys.executable, "train_perceptron.py"])
        if result.returncode != 0:
            print("❌ Training gagal!")
            sys.exit(1)
    else:
        print("✅ Model sudah ada, skip training!")

    # 3. Jalankan deteksi
    print()
    print("🎥 Membuka kamera...")
    print("   ▸ Tunjukkan HURUF KAPITAL ke dalam kotak emas")
    print("   ▸ Tekan Q untuk keluar")
    print()
    
    import subprocess
    subprocess.run([sys.executable, "detect_alfabet.py"])


if __name__ == "__main__":
    main()
