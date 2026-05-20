# 🔤 Perceptron Alfabet Detector

Aplikasi deteksi huruf alfabet **A–Z** secara real-time menggunakan webcam, dibangun dengan algoritma **Perceptron (Multi-Layer Perceptron)** dan **OpenCV**. Sistem mampu membedakan huruf **Vokal** dan **Konsonan** secara otomatis.


## ✨ Fitur

- 📷 Deteksi huruf **real-time** via webcam
- 🧠 Model **Perceptron (MLP)** dengan akurasi **97.40%**
- 🔴 Otomatis klasifikasi **Vokal** (A, E, I, O, U) dan 🔵 **Konsonan**
- 📊 Tampilan **Top 5 prediksi** dengan confidence bar
- 📝 **History log** huruf yang terdeteksi selama sesi
- 🖼️ Preview **threshold** gambar real-time
- 🎨 UI modern dengan panel info lengkap

---

## 🛠️ Teknologi

| Komponen | Library |
|---|---|
| Computer Vision | OpenCV |
| Machine Learning | scikit-learn (MLPClassifier) |
| Ekstraksi Fitur | HOG Descriptor + Pixel Features |
| Image Processing | Pillow (PIL) |
| Numerik | NumPy |

---

## 📁 Struktur Project

```
perceptron-alfabet-detector/
├── train_perceptron.py   # Script training model
├── detect_alfabet.py     # Program utama (kamera)
├── run.py                # Entry point otomatis
└── README.md
```

> File `.pkl` (model) **tidak di-push** ke repo. Generate sendiri dengan menjalankan `train_perceptron.py`.

---

## 🚀 Cara Menjalankan

### 1. Clone repo
```bash
git clone https://github.com/MunaFatinah/perceptron-alfabet-detector.giT
cd perceptron-alfabet-detector
```

### 2. Install dependencies
```bash
pip install opencv-python scikit-learn numpy pillow
```

### 3. Train model (cukup sekali)
```bash
python train_perceptron.py
```
Proses training memakan waktu **2–5 menit**. Model akan disimpan sebagai `perceptron_alfabet.pkl`.

### 4. Jalankan deteksi
```bash
python detect_alfabet.py
```

Atau gunakan entry point otomatis yang menggabungkan semua langkah:
```bash
python run.py
```

---

## 🎮 Cara Pakai

1. Tulis huruf kapital dengan **spidol hitam tebal** di **kertas putih**
2. Tunjukkan kertas ke kamera, masukkan huruf ke dalam **kotak biru putus-putus**
3. Pastikan pencahayaan cukup terang dan huruf kontras dengan background
4. Hasil deteksi muncul di atas kotak beserta persentase confidence

### Shortcut keyboard
| Tombol | Fungsi |
|---|---|
| `Q` / `Esc` | Keluar dari program |
| `C` | Clear history log huruf |

---

## 🧠 Cara Kerja (Algoritma)

### 1. Preprocessing
- Frame kamera di-crop sesuai area ROI (Region of Interest)
- Dikonversi ke grayscale → Gaussian Blur → Adaptive Threshold

### 2. Ekstraksi Fitur
- **Pixel features**: nilai piksel 28×28 = 784 dimensi
- **HOG (Histogram of Oriented Gradients)**: 324 dimensi
- Total: **1108 dimensi fitur** per gambar

### 3. Model Perceptron (MLP)
```
Input (1108) → Dense(512) → Dense(256) → Dense(128) → Output(26)
Aktivasi: ReLU | Optimizer: Adam | Early Stopping: ✓
```

### 4. Prediksi Stabil
Voting dari **12 frame terakhir** untuk menghindari prediksi yang berfluktuasi.

---

## 📊 Hasil Training

| Metrik | Nilai |
|---|---|
| Akurasi keseluruhan | **97.40%** |
| Jumlah kelas | 26 huruf (A–Z) |
| Dataset | 5.200 sampel sintetis |
| Dimensi fitur | 1.108 |

Huruf dengan akurasi 100%: **A, E, I, L, Q, X, Z**

---

## ⚠️ Tips untuk Hasil Terbaik

- Gunakan **spidol hitam tebal** (bukan pensil/pulpen tipis)
- Kertas harus **putih bersih** (bukan buram/abu-abu)
- Pastikan **cahaya dari depan** kertas, bukan dari belakang
- Huruf sebaiknya **kapital dan besar** memenuhi kotak
- Hindari bayangan tangan menutupi huruf

---

## 👩‍💻 Author

Dibuat sebagai tugas kuliah — implementasi algoritma **Perceptron** untuk pengenalan pola alfabet Latin A–Z.

---
