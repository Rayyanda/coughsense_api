import os
import warnings
import hashlib
import datetime
import io
import uuid
import sqlite3
import math

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

import numpy as np
import tensorflow as tf
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="TBC Cough Detection API",
    description="API deteksi batuk TBC dengan autentikasi pengguna",
    version="3.0.0"
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
TARGET_LENGTH  = input_details[0]["shape"][1]

with open(LABELS_PATH, "r") as f:
    raw_labels = [line.strip() for line in f.readlines()]
    labels = []
    for l in raw_labels:
        parts = l.split(" ", 1)
        if len(parts) == 2 and parts[0].isdigit():
            labels.append(parts[1])
        else:
            labels.append(l)

print(f"Labels loaded: {labels}")
print(f"Target length: {TARGET_LENGTH}")

# ─── Database ──────────────────────────────────────────────────────────────────
def get_conn():
    return sqlite3.connect("tbc.db")

def init_db():
    conn = get_conn()

    # Tabel users
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         TEXT PRIMARY KEY,
            nama       TEXT NOT NULL,
            username   TEXT NOT NULL UNIQUE,
            password   TEXT NOT NULL,
            usia       INTEGER,
            gender     TEXT,
            created_at TEXT
        )
    """)

    # Tabel predictions — tambah kolom user_id
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          TEXT PRIMARY KEY,
            user_id     TEXT,
            timestamp   TEXT,
            label       TEXT,
            confidence  REAL,
            all_scores  TEXT,
            mfcc_mean   TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()

init_db()

# ─── Helper password ───────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed

# ─── Pydantic models ───────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    nama:     str
    username: str
    password: str
    usia:     int
    gender:   str  # "Laki-laki" atau "Perempuan"

class LoginRequest(BaseModel):
    username: str
    password: str


def sanitize_float(v: float) -> float:
    """Ganti NaN/Inf dengan 0 agar JSON compliant."""
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v

# ─── MFCC Extraction ───────────────────────────────────────────────────────────
def extract_mfcc(y: np.ndarray) -> dict:
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

    if len(y) < TARGET_LENGTH:
        y = np.pad(y, (0, TARGET_LENGTH - len(y)))
    else:
        y = y[:TARGET_LENGTH]

    mfcc_info   = extract_mfcc(y)
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

    scores = {labels[i]: sanitize_float(float(output[i])) for i in range(len(labels))}
    predicted_label = labels[int(np.argmax(output))]
    confidence      = float(np.max(output))

    return {
        "label":      predicted_label,
        "confidence": round(confidence * 100, 2),
        "scores":     {k: round(v * 100, 2) for k, v in scores.items()},
    }

# ─── Warmup ───────────────────────────────────────────────────────────────────
def warmup_model():
    dummy = np.zeros((1, TARGET_LENGTH), dtype=np.float32)
    interpreter.set_tensor(input_details[0]["index"], dummy)
    interpreter.invoke()
    print("Model warmed up ✓")

warmup_model()

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "ok", "message": "TBC Cough Detection API v3 aktif"}

@app.get("/health")
def health():
    return {
        "status":  "healthy",
        "model":   MODEL_PATH,
        "labels":  labels,
        "version": "3.0.0",
    }

# ─── Auth ─────────────────────────────────────────────────────────────────────

@app.post("/register")
def register(data: RegisterRequest):
    """Daftar akun baru."""
    conn = get_conn()

    # Cek username sudah ada
    existing = conn.execute(
        "SELECT id FROM users WHERE username = ?", (data.username,)
    ).fetchone()

    if existing:
        conn.close()
        raise HTTPException(status_code=400, detail="Username sudah digunakan")

    # Validasi gender
    if data.gender not in ["Laki-laki", "Perempuan"]:
        conn.close()
        raise HTTPException(status_code=400, detail="Gender harus 'Laki-laki' atau 'Perempuan'")

    # Validasi usia
    if data.usia < 1 or data.usia > 120:
        conn.close()
        raise HTTPException(status_code=400, detail="Usia tidak valid")

    user_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            user_id,
            data.nama.strip(),
            data.username.strip().lower(),
            hash_password(data.password),
            data.usia,
            data.gender,
            datetime.datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()

    return {
        "message": "Registrasi berhasil",
        "user": {
            "id":       user_id,
            "nama":     data.nama,
            "username": data.username,
            "usia":     data.usia,
            "gender":   data.gender,
        },
    }

@app.post("/login")
def login(data: LoginRequest):
    """Login dengan username dan password."""
    conn = get_conn()
    row = conn.execute(
        "SELECT id, nama, username, password, usia, gender FROM users WHERE username = ?",
        (data.username.strip().lower(),),
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=401, detail="Username tidak ditemukan")

    if not verify_password(data.password, row[3]):
        raise HTTPException(status_code=401, detail="Password salah")

    return {
        "message": "Login berhasil",
        "user": {
            "id":       row[0],
            "nama":     row[1],
            "username": row[2],
            "usia":     row[4],
            "gender":   row[5],
        },
    }

# ─── Predict ──────────────────────────────────────────────────────────────────

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    user_id: str = "",
):
    """
    Endpoint prediksi TBC.
    - **file**    : file audio .wav
    - **user_id** : id pengguna (opsional, dari login)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Tidak ada file yang dikirim")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File terlalu besar (maks 10 MB)")

    features, mfcc_info = preprocess_audio(contents)
    result = run_inference(features)

    pred_id = str(uuid.uuid4())
    conn    = get_conn()
    conn.execute(
        "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            pred_id,
            user_id if user_id else None,
            datetime.datetime.now().isoformat(),
            result["label"],
            result["confidence"],
            str(result["scores"]),
            str(mfcc_info["mfcc_mean"].tolist()),
        ),
    )
    conn.commit()
    conn.close()

    return {
        "id":         pred_id,
        "timestamp":  datetime.datetime.now().isoformat(),
        "prediction": result["label"],
        "confidence": result["confidence"],
        "all_scores": result["scores"],
        "is_tbc":     result["label"] == "+TB",
        "mfcc_features": {
            "n_mfcc":      N_MFCC,
            "mfcc_mean":   [round(sanitize_float(float(v)), 4) for v in mfcc_info["mfcc_mean"]],
            "mfcc_std":    [round(sanitize_float(float(v)), 4) for v in mfcc_info["mfcc_std"]],
            "delta_mean":  [round(sanitize_float(float(v)), 4) for v in np.mean(mfcc_info["mfcc_delta"],  axis=1)],
            "delta2_mean": [round(sanitize_float(float(v)), 4) for v in np.mean(mfcc_info["mfcc_delta2"], axis=1)],
            "description": (
                f"Diekstraksi menggunakan metode MFCC dengan {N_MFCC} koefisien, "
                f"window 25ms, hop 10ms, sample rate {SAMPLE_RATE}Hz."
            ),
        },
    }

# ─── History ──────────────────────────────────────────────────────────────────

@app.get("/history/{user_id}")
def get_history(user_id: str, limit: int = 20):
    """Ambil riwayat prediksi milik user tertentu."""
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT id, timestamp, label, confidence
        FROM predictions
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()
    conn.close()

    return {
        "data": [
            {
                "id":         r[0],
                "timestamp":  r[1],
                "label":      r[2],
                "confidence": r[3],
            }
            for r in rows
        ]
    }

@app.delete("/history/{pred_id}")
def delete_prediction(pred_id: str):
    """Hapus satu record prediksi."""
    conn = get_conn()
    conn.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
    conn.commit()
    conn.close()
    return {"message": "Berhasil dihapus"}

# ─── Profile ──────────────────────────────────────────────────────────────────

@app.get("/profile/{user_id}")
def get_profile(user_id: str):
    """Ambil data profil pengguna."""
    conn = get_conn()
    row = conn.execute(
        "SELECT id, nama, username, usia, gender, created_at FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")

    return {
        "id":         row[0],
        "nama":       row[1],
        "username":   row[2],
        "usia":       row[3],
        "gender":     row[4],
        "created_at": row[5],
    }