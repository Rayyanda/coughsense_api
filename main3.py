import os
import warnings
# Sembunyikan warning TensorFlow yang tidak perlu
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import librosa
import io
import sqlite3
import datetime
import uuid

app = FastAPI(
    title="TBC Cough Detection API",
    description="API untuk mendeteksi batuk TBC menggunakan model Teachable Machine + MFCC",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Konfigurasi ───────────────────────────────────────────────────────────────
MODEL_PATH  = "model/model.tflite"
LABELS_PATH = "model/labels.txt"
SAMPLE_RATE = 16000
N_MFCC      = 40

# ─── Load model ────────────────────────────────────────────────────────────────
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
TARGET_LENGTH    = input_details[0]["shape"][1] 

with open(LABELS_PATH, "r") as f:
    raw_labels = [line.strip() for line in f.readlines()]
    labels = []
    for l in raw_labels:
        parts = l.split(" ", 1)
        if len(parts) == 2 and parts[0].isdigit():
            labels.append(parts[1])
        else:
            labels.append(l)

# ─── Database ──────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect("predictions.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id           TEXT PRIMARY KEY,
            timestamp    TEXT,
            label        TEXT,
            confidence   REAL,
            all_scores   TEXT,
            mfcc_mean    TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_prediction(pred_id, label, confidence, all_scores, mfcc_mean):
    conn = sqlite3.connect("predictions.db")
    conn.execute(
        "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
        (
            pred_id,
            datetime.datetime.now().isoformat(),
            label,
            confidence,
            str(all_scores),
            str(mfcc_mean),
        ),
    )
    conn.commit()
    conn.close()

# ─── MFCC Extraction (selalu dijalankan) ───────────────────────────────────────
def extract_mfcc(y: np.ndarray) -> dict:
    """
    Ekstraksi fitur MFCC secara eksplisit menggunakan librosa.

    Parameter:
    - n_mfcc     : 40 koefisien
    - window     : 25ms (win_length=400 pada 16kHz)
    - hop        : 10ms (hop_length=160 pada 16kHz)
    - n_fft      : 512

    Menghasilkan:
    - mfcc_matrix : matriks mentah [40 x n_frames]
    - mfcc_mean   : rata-rata per koefisien → representasi ringkas audio
    - mfcc_std    : standar deviasi per koefisien → variasi temporal
    - mfcc_min    : nilai minimum per koefisien
    - mfcc_max    : nilai maksimum per koefisien
    - mfcc_delta  : delta MFCC (turunan pertama — perubahan antar frame)
    - mfcc_delta2 : delta-delta MFCC (turunan kedua)
    """
    mfcc_matrix = librosa.feature.mfcc(
        y=y,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=512,
        hop_length=160,
        win_length=400,
    )

    mfcc_delta  = librosa.feature.delta(mfcc_matrix)
    mfcc_delta2 = librosa.feature.delta(mfcc_matrix, order=2)

    return {
        "mfcc_matrix":  mfcc_matrix,
        "mfcc_mean":    np.mean(mfcc_matrix, axis=1),
        "mfcc_std":     np.std(mfcc_matrix,  axis=1),
        "mfcc_min":     np.min(mfcc_matrix,  axis=1),
        "mfcc_max":     np.max(mfcc_matrix,  axis=1),
        "mfcc_delta":   mfcc_delta,
        "mfcc_delta2":  mfcc_delta2,
    }

# ─── Audio Preprocessing ───────────────────────────────────────────────────────
def preprocess_audio(audio_bytes: bytes) -> tuple:
    audio_io = io.BytesIO(audio_bytes)

    try:
        y, sr = librosa.load(audio_io, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Gagal membaca audio: {str(e)}")

    # Gunakan TARGET_LENGTH dari model langsung
    if len(y) < TARGET_LENGTH:
        y = np.pad(y, (0, TARGET_LENGTH - len(y)))
    else:
        y = y[:TARGET_LENGTH]

    mfcc_info = extract_mfcc(y)

    input_shape = input_details[0]["shape"]

    if len(input_shape) == 2 and input_shape[1] != N_MFCC:
        features = y.reshape(1, -1).astype(np.float32)
    elif len(input_shape) == 3 and input_shape[2] == 1:
        features = y.reshape(1, -1, 1).astype(np.float32)
    elif len(input_shape) == 2 and input_shape[1] == N_MFCC:
        features = mfcc_info["mfcc_mean"].reshape(1, -1).astype(np.float32)
    else:
        feature_vector = np.concatenate([
            mfcc_info["mfcc_mean"],
            mfcc_info["mfcc_std"],
            np.mean(mfcc_info["mfcc_delta"], axis=1),
        ])
        features = feature_vector.reshape(1, -1).astype(np.float32)

    return features, mfcc_info

# ─── Inference ────────────────────────────────────────────────────────────────
def run_inference(features: np.ndarray) -> dict:
    interpreter.set_tensor(input_details[0]["index"], features)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    scores          = {labels[i]: float(output[i]) for i in range(len(labels))}
    predicted_label = labels[int(np.argmax(output))]
    confidence      = float(np.max(output))
    print(f"Debug Inference: {scores}, Predicted: {predicted_label} ({confidence:.4f})")
    return {
        "label":      predicted_label,
        "confidence": round(confidence * 100, 2),
        "scores":     {k: round(v * 100, 2) for k, v in scores.items()},
    }

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "TBC Cough Detection API v2 aktif"}

@app.get("/health")
def health():
    return {
        "status":       "healthy",
        "model":        MODEL_PATH,
        "labels":       labels,
        "mfcc_config": {
            "n_mfcc":      N_MFCC,
            "sample_rate": SAMPLE_RATE,
            "duration_s":  TARGET_LENGTH / SAMPLE_RATE,
            "hop_length":  160,
            "win_length":  400,
            "n_fft":       512,
        },
        "version": "2.0.0",
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint prediksi TBC.

    Pipeline lengkap:
    1. Terima file audio (.wav / .mp3 / .m4a / .ogg)
    2. Resample → 16kHz mono
    3. **Ekstraksi MFCC 40 koefisien** (window 25ms, hop 10ms)
       + Delta MFCC + Delta-delta MFCC
    4. Inferensi model TFLite
    5. Kembalikan hasil prediksi + detail fitur MFCC
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Tidak ada file yang dikirim")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File terlalu besar (maks 10 MB)")

    features, mfcc_info = preprocess_audio(contents)
    result = run_inference(features)

    pred_id = str(uuid.uuid4())
    save_prediction(
        pred_id,
        result["label"],
        result["confidence"],
        result["scores"],
        mfcc_info["mfcc_mean"].tolist(),
    )
    
    return {
        "id":         pred_id,
        "timestamp":  datetime.datetime.now().isoformat(),
        "prediction": result["label"],
        "confidence": result["confidence"],
        "all_scores": result["scores"],
        # "is_tbc":     result["label"].upper() in ["TBC", "POSITIF", "TB"],
        # "is_tbc":     result["label"] == '+TB',
        # Ganti baris is_tbc di endpoint /predict
        "is_tbc": "+" in result["label"] or "tbc" in result["label"].lower() or "positif" in result["label"].lower(),

        # ── Detail MFCC — ditampilkan di Flutter & dijelaskan ke juri ─────────
        "mfcc_features": {
            "n_mfcc":      N_MFCC,
            "mfcc_mean":   [round(float(v), 4) for v in mfcc_info["mfcc_mean"]],
            "mfcc_std":    [round(float(v), 4) for v in mfcc_info["mfcc_std"]],
            "mfcc_min":    [round(float(v), 4) for v in mfcc_info["mfcc_min"]],
            "mfcc_max":    [round(float(v), 4) for v in mfcc_info["mfcc_max"]],
            "delta_mean":  [round(float(v), 4) for v in np.mean(mfcc_info["mfcc_delta"],  axis=1)],
            "delta2_mean": [round(float(v), 4) for v in np.mean(mfcc_info["mfcc_delta2"], axis=1)],
            "description": (
                f"Diekstraksi menggunakan metode MFCC dengan {N_MFCC} koefisien, "
                f"window 25ms, hop 10ms, sample rate {SAMPLE_RATE}Hz, "
                f"durasi audio {TARGET_LENGTH / SAMPLE_RATE} detik."
            ),
        },
    }

@app.get("/history")
def get_history(limit: int = 20):
    """Ambil riwayat prediksi terakhir."""
    conn = sqlite3.connect("predictions.db")
    rows = conn.execute(
        "SELECT id, timestamp, label, confidence FROM predictions ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return {
        "data": [
            {"id": r[0], "timestamp": r[1], "label": r[2], "confidence": r[3]}
            for r in rows
        ]
    }

@app.delete("/history/{pred_id}")
def delete_prediction(pred_id: str):
    """Hapus satu record prediksi."""
    conn = sqlite3.connect("predictions.db")
    conn.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
    conn.commit()
    conn.close()
    return {"message": "Berhasil dihapus"}