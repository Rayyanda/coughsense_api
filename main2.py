from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
# import tflite_runtime.interpreter as tflite
import librosa
import io
import soundfile as sf
import sqlite3
import datetime
import os
import uuid

app = FastAPI(
    title="TBC Cough Detection API",
    description="API untuk mendeteksi batuk TBC menggunakan model Teachable Machine",
    version="1.0.0"
)

# CORS - izinkan akses dari Flutter / mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Konfigurasi ───────────────────────────────────────────────────────────────
MODEL_PATH  = "model/model.tflite"
LABELS_PATH = "model/labels.txt"
SAMPLE_RATE = 16000      # Hz — sesuaikan dengan setting Teachable Machine
DURATION    = 3          # detik — window audio yang digunakan TM (biasanya 3 s)
N_MFCC      = 40         # jumlah MFCC features

# ─── Load model TFLite ─────────────────────────────────────────────────────────
# interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)  # fallback ke tf.lite jika tflite_runtime bermasalah
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ─── Database setup ────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect("predictions.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          TEXT PRIMARY KEY,
            timestamp   TEXT,
            label       TEXT,
            confidence  REAL,
            all_scores  TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_prediction(pred_id: str, label: str, confidence: float, all_scores: dict):
    conn = sqlite3.connect("predictions.db")
    conn.execute(
        "INSERT INTO predictions VALUES (?, ?, ?, ?, ?)",
        (pred_id, datetime.datetime.now().isoformat(), label, confidence, str(all_scores))
    )
    conn.commit()
    conn.close()

# ─── Audio preprocessing ───────────────────────────────────────────────────────
def preprocess_audio(audio_bytes: bytes) -> np.ndarray:
    """
    Konversi audio bytes → MFCC features sesuai format input Teachable Machine.
    Teachable Machine Audio menggunakan representasi spectrogram / waveform
    dengan panjang tetap (SAMPLE_RATE * DURATION).
    """
    audio_io = io.BytesIO(audio_bytes)

    # Baca audio (mendukung .wav, .mp3, .m4a, .ogg, dll.)
    try:
        y, sr = librosa.load(audio_io, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Gagal membaca audio: {str(e)}")

    # Pastikan panjang tepat SAMPLE_RATE * DURATION sampel
    target_length = SAMPLE_RATE * DURATION
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    # Teachable Machine Audio model menerima raw waveform float32 ternormalisasi
    # Shape: [1, target_length] atau [1, target_length, 1] — cek input_details
    input_shape = input_details[0]["shape"]

    if len(input_shape) == 2:
        # Shape: [1, samples]
        features = y.reshape(1, -1).astype(np.float32)
    elif len(input_shape) == 3 and input_shape[2] == 1:
        # Shape: [1, samples, 1]
        features = y.reshape(1, -1, 1).astype(np.float32)
    else:
        # Fallback: MFCC jika model custom menggunakan MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        mfcc = np.mean(mfcc.T, axis=0)
        features = mfcc.reshape(1, -1).astype(np.float32)

    return features

# ─── Inference ────────────────────────────────────────────────────────────────
def run_inference(features: np.ndarray) -> dict:
    interpreter.set_tensor(input_details[0]["index"], features)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    scores = {labels[i]: float(output[i]) for i in range(len(labels))}
    predicted_label = labels[int(np.argmax(output))]
    confidence = float(np.max(output))

    return {
        "label":      predicted_label,
        "confidence": round(confidence * 100, 2),
        "scores":     {k: round(v * 100, 2) for k, v in scores.items()}
    }

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "TBC Cough Detection API aktif"}

@app.get("/health")
def health():
    return {
        "status":  "healthy",
        "model":   MODEL_PATH,
        "labels":  labels,
        "version": "1.0.0"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint utama: terima file audio, kembalikan prediksi TBC / Non-TBC.

    - **file**: File audio (.wav / .mp3 / .m4a / .ogg), durasi ~3 detik
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Tidak ada file yang dikirim")

    # Batasi ukuran file (maks 10 MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File terlalu besar (maks 10 MB)")

    features = preprocess_audio(contents)
    result   = run_inference(features)

    pred_id  = str(uuid.uuid4())
    save_prediction(pred_id, result["label"], result["confidence"], result["scores"])

    return {
        "id":           pred_id,
        "timestamp":    datetime.datetime.now().isoformat(),
        "prediction":   result["label"],
        "confidence":   result["confidence"],
        "all_scores":   result["scores"],
        "is_tbc":       result["label"].upper() in ["TBC", "POSITIF", "TB"]
    }

@app.get("/history")
def get_history(limit: int = 20):
    """Ambil riwayat prediksi terakhir."""
    conn = sqlite3.connect("predictions.db")
    rows = conn.execute(
        "SELECT id, timestamp, label, confidence FROM predictions ORDER BY timestamp DESC LIMIT ?",
        (limit,)
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