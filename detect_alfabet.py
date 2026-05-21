"""
╔══════════════════════════════════════════════════╗
║   PERCEPTRON ALFABET DETECTOR                    ║
║   Deteksi Huruf A-Z via Kamera Real-Time         ║
║   Menggunakan: OpenCV + Perceptron (MLP)         ║
╚══════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import pickle
import os
import time
from collections import deque, Counter

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
MODEL_PATH  = "perceptron_alfabet.pkl"
SCALER_PATH = "scaler_alfabet.pkl"
IMG_SIZE    = 28
LABELS      = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
VOKAL       = set("AEIOU")

# Warna (BGR)
C_BG        = (15, 15, 25)          # background gelap navy
C_ACCENT    = (255, 180, 50)        # emas
C_VOKAL     = (80, 200, 255)        # biru cyan (vokal)
C_KONSONAN  = (100, 255, 170)       # hijau mint (konsonan)
C_WHITE     = (240, 240, 240)
C_GRAY      = (120, 120, 140)
C_PANEL     = (28, 28, 45)
C_BOX       = (255, 180, 50)        # kotak ROI

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        print("❌ Model belum ada! Jalankan dulu: python train_perceptron.py")
        exit(1)
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Model Perceptron berhasil dimuat!")
    return model, scaler


# ─────────────────────────────────────────────
# EKSTRAKSI FITUR (sama seperti training)
# ─────────────────────────────────────────────
def extract_hog_features(img_flat):
    img = img_flat.reshape(IMG_SIZE, IMG_SIZE)
    img_uint8 = (img * 255).astype(np.uint8)
    winSize    = (IMG_SIZE, IMG_SIZE)
    blockSize  = (IMG_SIZE//2, IMG_SIZE//2)
    blockStride= (IMG_SIZE//4, IMG_SIZE//4)
    cellSize   = (IMG_SIZE//4, IMG_SIZE//4)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    return hog.compute(img_uint8).flatten()

def preprocess_roi(roi):
    """Ubah ROI kamera menjadi fitur untuk prediksi."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Otsu threshold - otomatis nyesuain kecerahan
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Resize ke IMG_SIZE
    resized = cv2.resize(thresh, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype(np.float32) / 255.0

    pixel_feat = normalized.flatten()
    hog_feat   = extract_hog_features(pixel_feat)
    return np.concatenate([pixel_feat, hog_feat]), thresh


# ─────────────────────────────────────────────
# HELPER DRAW FUNCTIONS
# ─────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1, alpha=1.0):
    """Gambar rectangle dengan sudut membulat."""
    x1, y1 = pt1
    x2, y2 = pt2
    overlay = img.copy()
    cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, thickness)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, thickness)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    else:
        img[:] = overlay

def put_text_centered(img, text, center_x, y, font, scale, color, thickness=1):
    """Tulis teks rata tengah."""
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, (center_x - tw//2, y), font, scale, color, thickness, cv2.LINE_AA)

def draw_confidence_bar(img, x, y, w, h, value, color, bg_color=(40,40,60)):
    """Bar horizontal confidence."""
    cv2.rectangle(img, (x, y), (x+w, y+h), bg_color, -1)
    fill_w = int(w * value)
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x+fill_w, y+h), color, -1)
    # Border
    cv2.rectangle(img, (x, y), (x+w, y+h), (60, 60, 80), 1)


# ─────────────────────────────────────────────
# MAIN PROGRAM
# ─────────────────────────────────────────────
def main():
    print("\n╔══════════════════════════════════════╗")
    print("║  PERCEPTRON ALFABET DETECTOR  🔤     ║")
    print("╚══════════════════════════════════════╝")
    
    model, scaler = load_model()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Kamera tidak bisa dibuka!")
        exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX
    FONT_MONO  = cv2.FONT_HERSHEY_COMPLEX_SMALL
    
    # State
    prediction_history = deque(maxlen=12)
    confidence_history = deque(maxlen=12)
    stable_letter      = "?"
    stable_conf        = 0.0
    letter_timer       = 0
    fps_counter        = deque(maxlen=30)
    last_time          = time.time()
    detected_letters   = []          # log huruf yang pernah terdeteksi
    
    # ROI box di tengah frame
    ROI_SIZE = 200
    
    print("\n📷 Kamera aktif!")
    print("   ▸ Tunjukkan HURUF ke dalam kotak emas")
    print("   ▸ Tekan  Q  untuk keluar")
    print("   ▸ Tekan  C  untuk clear log huruf\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)   # mirror
        H, W  = frame.shape[:2]
        
        # FPS
        now = time.time()
        fps_counter.append(1.0 / max(now - last_time, 0.001))
        last_time = now
        fps = np.mean(fps_counter)
        
        # ── BUAT CANVAS OUTPUT ──────────────────────────────────
        # Panel kiri (info) : 320px | Kamera: sisa | (total lebar 1280)
        PANEL_W = 340
        cam_w   = W
        canvas_w = cam_w + PANEL_W
        canvas   = np.full((H, canvas_w, 3), C_BG, dtype=np.uint8)
        
        # Letakkan frame kamera
        canvas[:, :cam_w] = frame
        
        # ── ROI (kotak emas di tengah kamera) ───────────────────
        cx = cam_w // 2
        cy = H // 2
        rx1, ry1 = cx - ROI_SIZE//2, cy - ROI_SIZE//2
        rx2, ry2 = cx + ROI_SIZE//2, cy + ROI_SIZE//2
        
        roi = frame[ry1:ry2, rx1:rx2]
        
        # Prediksi
        letter      = "?"
        conf        = 0.0
        proba_all   = np.zeros(26)
        thresh_preview = None
        
        if roi.size > 0:
            features, thresh_preview = preprocess_roi(roi)
            features_scaled = scaler.transform([features])
            
            pred_idx = model.predict(features_scaled)[0]
            letter   = LABELS[pred_idx]
            
            # Probabilitas (pakai decision function → softmax)
            if hasattr(model, 'predict_proba'):
                proba_all = model.predict_proba(features_scaled)[0]
            else:
                df = model.decision_function(features_scaled)[0]
                e  = np.exp(df - df.max())
                proba_all = e / e.sum()
            conf = proba_all[pred_idx]
            
            prediction_history.append(pred_idx)
            confidence_history.append(conf)
            
            # Stable prediction (voting)
            if len(prediction_history) >= 6:
                votes = Counter(prediction_history)
                stable_idx   = votes.most_common(1)[0][0]
                stable_letter = LABELS[stable_idx]
                stable_conf   = float(np.mean(confidence_history))
                
                # Auto-log jika confidence tinggi & stabil
                if stable_conf > 0.70 and time.time() - letter_timer > 1.5:
                    if not detected_letters or detected_letters[-1] != stable_letter:
                        detected_letters.append(stable_letter)
                        if len(detected_letters) > 20:
                            detected_letters.pop(0)
                        letter_timer = time.time()
        
        is_vokal = stable_letter in VOKAL
        accent   = C_VOKAL if is_vokal else C_KONSONAN
        
        # ── DRAW PANEL KANAN ─────────────────────────────────────
        px = cam_w + 16  # left edge panel
        pw = PANEL_W - 24
        
        # Background panel
        canvas[:, cam_w:] = C_PANEL
        # Garis separator
        cv2.line(canvas, (cam_w, 0), (cam_w, H), C_ACCENT, 2)
        
        # ── HEADER ──────────────────────────────────────────────
        draw_rounded_rect(canvas, (cam_w+8, 10), (cam_w+PANEL_W-8, 60), C_ACCENT, radius=8, alpha=1.0)
        put_text_centered(canvas, "PERCEPTRON DETEKTOR", cam_w + PANEL_W//2, 32, FONT_BOLD, 0.45, C_BG, 1)
        put_text_centered(canvas, "Alfabet A-Z | Vokal & Konsonan", cam_w + PANEL_W//2, 52, FONT_MONO, 0.42, C_BG, 1)
        
        # ── HURUF UTAMA ──────────────────────────────────────────
        draw_rounded_rect(canvas, (px, 72), (px+pw, 210), (30, 32, 50), radius=12)
        
        # Label vokal/konsonan
        jenis_label = "HURUF VOKAL" if is_vokal else "HURUF KONSONAN"
        put_text_centered(canvas, jenis_label, cam_w+PANEL_W//2, 96, FONT, 0.50, accent, 1)
        
        # Huruf besar
        (lw, lh), _ = cv2.getTextSize(stable_letter, FONT_BOLD, 6.0, 4)
        lx = cam_w + PANEL_W//2 - lw//2
        cv2.putText(canvas, stable_letter, (lx, 180), FONT_BOLD, 6.0, accent, 4, cv2.LINE_AA)
        
        # Glow effect (outline tipis)
        cv2.putText(canvas, stable_letter, (lx, 180), FONT_BOLD, 6.0, 
                    tuple(max(0, c-80) for c in accent), 2, cv2.LINE_AA)
        
        # Confidence bar
        bar_y = 218
        cv2.putText(canvas, "CONFIDENCE", (px, bar_y-5), FONT_MONO, 0.42, C_GRAY, 1)
        conf_label = f"{stable_conf*100:.1f}%"
        cv2.putText(canvas, conf_label, (px+pw-60, bar_y-5), FONT_MONO, 0.42, accent, 1)
        draw_confidence_bar(canvas, px, bar_y, pw, 14, stable_conf, accent)
        
        # ── TOP-5 PREDIKSI ────────────────────────────────────────
        top5_y = 250
        cv2.putText(canvas, "TOP 5 PREDIKSI", (px, top5_y), FONT_MONO, 0.44, C_GRAY, 1)
        top5_y += 18
        
        top5_idx = np.argsort(proba_all)[::-1][:5]
        for rank, idx in enumerate(top5_idx):
            l   = LABELS[idx]
            p   = proba_all[idx]
            col = accent if rank == 0 else C_GRAY
            row_y = top5_y + rank * 32
            
            draw_rounded_rect(canvas, (px, row_y), (px+pw, row_y+26),
                              (35, 35, 55) if rank == 0 else (25, 25, 40), radius=5)
            
            lbl = "🔴V" if l in VOKAL else "🔵K"
            cv2.putText(canvas, l, (px+8, row_y+19), FONT_BOLD, 0.65, col, 1, cv2.LINE_AA)
            cv2.putText(canvas, ("V" if l in VOKAL else "K"), (px+28, row_y+19), FONT_MONO, 0.38, 
                        C_VOKAL if l in VOKAL else C_KONSONAN, 1)
            
            bar_w = int((pw - 65) * p)
            cv2.rectangle(canvas, (px+50, row_y+8), (px+pw-40, row_y+18), (40,40,60), -1)
            if bar_w > 0:
                cv2.rectangle(canvas, (px+50, row_y+8), (px+50+bar_w, row_y+18), col, -1)
            cv2.putText(canvas, f"{p*100:.0f}%", (px+pw-36, row_y+19), FONT_MONO, 0.40, col, 1)
        
        # ── LOG HURUF TERDETEKSI ──────────────────────────────────
        log_y = top5_y + 5*32 + 14
        cv2.line(canvas, (px, log_y), (px+pw, log_y), (50, 50, 70), 1)
        log_y += 14
        cv2.putText(canvas, "HISTORY DETEKSI  (C=clear)", (px, log_y), FONT_MONO, 0.38, C_GRAY, 1)
        log_y += 18
        
        # Tampilkan huruf-huruf yang pernah dideteksi
        letters_str = " ".join(detected_letters[-14:])
        # Wrap jika panjang
        words = detected_letters[-14:]
        line1 = " ".join(words[:7])
        line2 = " ".join(words[7:])
        cv2.putText(canvas, line1, (px, log_y+16), FONT_BOLD, 0.60, C_WHITE, 1, cv2.LINE_AA)
        cv2.putText(canvas, line2, (px, log_y+36), FONT_BOLD, 0.60, C_WHITE, 1, cv2.LINE_AA)
        
        # Statistik vokal/konsonan
        nv = sum(1 for l in detected_letters if l in VOKAL)
        nk = len(detected_letters) - nv
        stat_y = log_y + 58
        cv2.putText(canvas, f"Vokal: {nv}   Konsonan: {nk}   Total: {len(detected_letters)}",
                    (px, stat_y), FONT_MONO, 0.40, C_GRAY, 1)
        
        # ── FPS & INFO ────────────────────────────────────────────
        cv2.putText(canvas, f"FPS: {fps:.0f}", (px, H-48), FONT_MONO, 0.42, C_GRAY, 1)
        cv2.putText(canvas, "Q=Keluar  C=Clear log", (px, H-28), FONT_MONO, 0.38, C_GRAY, 1)
        
        # ── ANOTASI DI FRAME KAMERA ─────────────────────────────
        # Preview threshold (pojok kiri bawah)
        if thresh_preview is not None:
            prev_size = 100
            prev = cv2.resize(thresh_preview, (prev_size, prev_size))
            prev_bgr = cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR)
            canvas[H-prev_size-10:H-10, 10:10+prev_size] = prev_bgr
            cv2.rectangle(canvas, (10, H-prev_size-10), (10+prev_size, H-10), C_ACCENT, 1)
            cv2.putText(canvas, "THRESH", (12, H-prev_size-14), FONT_MONO, 0.38, C_ACCENT, 1)
        
        # Kotak ROI emas animasi
        thickness = 2
        dash_len  = 18
        gap_len   = 8
        t = int(time.time() * 60) % (dash_len + gap_len)
        # Gambar garis putus-putus dengan animasi
        for side in range(4):
            if side == 0:   pts = [(rx1+i, ry1) for i in range(0, ROI_SIZE)]
            elif side == 1: pts = [(rx2, ry1+i) for i in range(0, ROI_SIZE)]
            elif side == 2: pts = [(rx2-i, ry2) for i in range(0, ROI_SIZE)]
            else:           pts = [(rx1, ry2-i) for i in range(0, ROI_SIZE)]
            
            offset = (t + side * (dash_len + gap_len)) % (dash_len + gap_len)
            i = offset
            while i < len(pts):
                p1 = pts[i]
                p2 = pts[min(i+dash_len-1, len(pts)-1)]
                cv2.line(canvas, p1, p2, C_BOX, 2, cv2.LINE_AA)
                i += dash_len + gap_len
        
        # Label di atas ROI
        label_str = f"  {stable_letter}  "
        (lbw, lbh), _ = cv2.getTextSize(label_str, FONT_BOLD, 0.9, 2)
        lbx = cx - lbw//2
        lby = ry1 - 12
        draw_rounded_rect(canvas, (lbx-6, lby-lbh-4), (lbx+lbw+6, lby+4), accent, radius=5)
        cv2.putText(canvas, label_str, (lbx, lby), FONT_BOLD, 0.9, C_BG, 2, cv2.LINE_AA)
        
        # Confidence di bawah ROI
        conf_txt = f"{stable_conf*100:.0f}% {'(VOKAL)' if is_vokal else '(KONSONAN)'}"
        put_text_centered(canvas, conf_txt, cx, ry2+22, FONT_MONO, 0.52, accent, 1)
        
        # Instruksi
        put_text_centered(canvas, "Tunjukkan huruf ke dalam kotak", 
                          cam_w//2, H-18, FONT_MONO, 0.45, C_GRAY, 1)
        
        # ── TAMPILKAN ─────────────────────────────────────────────
        cv2.imshow("Perceptron Alfabet Detector | Tekan Q untuk keluar", canvas)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            detected_letters.clear()
            prediction_history.clear()
            stable_letter = "?"
            stable_conf   = 0.0
            print("🗑️  Log huruf dihapus.")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n╔══════════════════════════════════════╗")
    print("║  Terima kasih! Program selesai. 👋    ║")
    print("╚══════════════════════════════════════╝")
    if detected_letters:
        print(f"  Huruf terdeteksi sesi ini: {' '.join(detected_letters)}")


if __name__ == "__main__":
    main()