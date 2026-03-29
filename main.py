import os
import warnings
import hashlib
import datetime
import io
import uuid
import sqlite3
import math
from typing import Optional

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

import numpy as np
import tensorflow as tf
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

app = FastAPI(
    title="TBC Cough Detection API",
    description="API deteksi batuk TBC dengan autentikasi pengguna",
    version="3.1.0"
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

VALID_GENDERS = ["Laki-laki", "Perempuan", "Male", "Female"]

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
    conn = sqlite3.connect("tbc.db")
    conn.row_factory = sqlite3.Row   # akses kolom by name
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    conn = get_conn()

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

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          TEXT PRIMARY KEY,
            user_id     TEXT,
            timestamp   TEXT,
            label       TEXT,
            confidence  REAL,
            all_scores  TEXT,
            mfcc_mean   TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
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
    gender:   str

    @field_validator("nama")
    @classmethod
    def nama_not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Nama tidak boleh kosong")
        return v

    @field_validator("username")
    @classmethod
    def username_valid(cls, v):
        v = v.strip().lower()
        if not v:
            raise ValueError("Username tidak boleh kosong")
        if len(v) < 3:
            raise ValueError("Username minimal 3 karakter")
        if not v.replace("_", "").replace(".", "").isalnum():
            raise ValueError("Username hanya boleh huruf, angka, titik, atau underscore")
        return v

    @field_validator("password")
    @classmethod
    def password_valid(cls, v):
        if len(v) < 6:
            raise ValueError("Password minimal 6 karakter")
        return v

    @field_validator("usia")
    @classmethod
    def usia_valid(cls, v):
        if v < 1 or v > 120:
            raise ValueError("Usia tidak valid (1–120)")
        return v

    @field_validator("gender")
    @classmethod
    def gender_valid(cls, v):
        if v not in VALID_GENDERS:
            raise ValueError(f"Gender harus salah satu dari: {', '.join(VALID_GENDERS)}")
        return v


class LoginRequest(BaseModel):
    username: str
    password: str


class UpdateProfileRequest(BaseModel):
    nama:         Optional[str] = None
    usia:         Optional[int] = None
    gender:       Optional[str] = None
    old_password: Optional[str] = None   # wajib jika ganti password
    new_password: Optional[str] = None

    @field_validator("nama")
    @classmethod
    def nama_not_empty(cls, v):
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("Nama tidak boleh kosong")
        return v

    @field_validator("usia")
    @classmethod
    def usia_valid(cls, v):
        if v is not None and (v < 1 or v > 120):
            raise ValueError("Usia tidak valid (1–120)")
        return v

    @field_validator("gender")
    @classmethod
    def gender_valid(cls, v):
        if v is not None and v not in VALID_GENDERS:
            raise ValueError(f"Gender harus salah satu dari: {', '.join(VALID_GENDERS)}")
        return v

    @field_validator("new_password")
    @classmethod
    def new_password_valid(cls, v):
        if v is not None and len(v) < 6:
            raise ValueError("Password baru minimal 6 karakter")
        return v


# ─── Helper ────────────────────────────────────────────────────────────────────
def sanitize_float(v: float) -> float:
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v

def user_row_to_dict(row) -> dict:
    return {
        "id":       row["id"],
        "nama":     row["nama"],
        "username": row["username"],
        "usia":     row["usia"],
        "gender":   row["gender"],
    }

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

    if len(y) == 0:
        raise HTTPException(status_code=422, detail="File audio kosong atau tidak valid")

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

# ─── Inference ─────────────────────────────────────────────────────────────────
def run_inference(features: np.ndarray) -> dict:
    interpreter.set_tensor(input_details[0]["index"], features)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    scores          = {labels[i]: sanitize_float(float(output[i])) for i in range(len(labels))}
    predicted_label = labels[int(np.argmax(output))]
    confidence      = float(np.max(output))

    return {
        "label":      predicted_label,
        "confidence": round(confidence * 100, 2),
        "scores":     {k: round(v * 100, 2) for k, v in scores.items()},
    }

# ─── Warmup ────────────────────────────────────────────────────────────────────
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
    return {"status": "ok", "message": "TBC Cough Detection API v3.1 aktif"}

@app.get("/health")
def health():
    return {
        "status":  "healthy",
        "model":   MODEL_PATH,
        "labels":  labels,
        "version": "3.1.0",
    }


# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/register", status_code=201)
def register(data: RegisterRequest):
    """Daftar akun baru."""
    conn = get_conn()

    existing = conn.execute(
        "SELECT id FROM users WHERE username = ?", (data.username,)
    ).fetchone()

    if existing:
        conn.close()
        raise HTTPException(status_code=400, detail="Username sudah digunakan")

    user_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            user_id,
            data.nama.strip(),
            data.username,
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
            "nama":     data.nama.strip(),
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

    if not verify_password(data.password, row["password"]):
        raise HTTPException(status_code=401, detail="Password salah")

    return {
        "message": "Login berhasil",
        "user": {
            "id":       row["id"],
            "nama":     row["nama"],
            "username": row["username"],
            "usia":     row["usia"],
            "gender":   row["gender"],
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# PROFILE
# ══════════════════════════════════════════════════════════════════════════════

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
        "id":         row["id"],
        "nama":       row["nama"],
        "username":   row["username"],
        "usia":       row["usia"],
        "gender":     row["gender"],
        "created_at": row["created_at"],
    }


@app.put("/profile/{user_id}")
def update_profile(user_id: str, data: UpdateProfileRequest):
    """
    Update profil pengguna.
    - Kirim hanya field yang ingin diubah (partial update).
    - Untuk ganti password, sertakan `old_password` dan `new_password`.
    """
    conn = get_conn()
    row = conn.execute(
        "SELECT id, nama, username, password, usia, gender FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="User tidak ditemukan")

    # Ambil nilai saat ini, ganti hanya yang dikirim
    new_nama   = data.nama   if data.nama   is not None else row["nama"]
    new_usia   = data.usia   if data.usia   is not None else row["usia"]
    new_gender = data.gender if data.gender is not None else row["gender"]
    new_hash   = row["password"]  # default tetap

    # Ganti password jika diminta
    if data.new_password is not None:
        if data.old_password is None:
            conn.close()
            raise HTTPException(
                status_code=400,
                detail="Masukkan password lama untuk mengganti password"
            )
        if not verify_password(data.old_password, row["password"]):
            conn.close()
            raise HTTPException(status_code=401, detail="Password lama salah")
        new_hash = hash_password(data.new_password)

    conn.execute(
        "UPDATE users SET nama = ?, usia = ?, gender = ?, password = ? WHERE id = ?",
        (new_nama, new_usia, new_gender, new_hash, user_id),
    )
    conn.commit()

    # Ambil ulang data terbaru
    updated = conn.execute(
        "SELECT id, nama, username, usia, gender, created_at FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    conn.close()

    return {
        "message": "Profil berhasil diperbarui",
        "user": {
            "id":         updated["id"],
            "nama":       updated["nama"],
            "username":   updated["username"],
            "usia":       updated["usia"],
            "gender":     updated["gender"],
            "created_at": updated["created_at"],
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    user_id: str = Query(default=""),
):
    """
    Endpoint prediksi TBC.
    - **file**    : file audio .wav
    - **user_id** : id pengguna (opsional, dari login)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Tidak ada file yang dikirim")

    # Validasi ekstensi
    allowed_exts = {".wav", ".mp3", ".ogg", ".m4a", ".flac"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=415,
            detail=f"Format file tidak didukung. Gunakan: {', '.join(allowed_exts)}"
        )

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="File kosong")
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File terlalu besar (maks 10 MB)")

    features, mfcc_info = preprocess_audio(contents)
    result = run_inference(features)

    pred_id   = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()

    conn = get_conn()

    # Validasi user_id jika disertakan
    if user_id:
        user_exists = conn.execute(
            "SELECT id FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if not user_exists:
            user_id = ""   # abaikan user_id tidak valid, tetap simpan prediksi

    conn.execute(
        "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            pred_id,
            user_id if user_id else None,
            timestamp,
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
        "timestamp":  timestamp,
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


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/history/{user_id}")
def get_history(user_id: str, limit: int = Query(default=20, ge=1, le=100)):
    """
    Ambil riwayat prediksi milik user.
    - **limit**: jumlah data (1–100, default 20)
    """
    conn = get_conn()

    # Pastikan user ada
    user = conn.execute(
        "SELECT id FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="User tidak ditemukan")

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
        "total": len(rows),
        "data": [
            {
                "id":         r["id"],
                "timestamp":  r["timestamp"],
                "label":      r["label"],
                "confidence": r["confidence"],
            }
            for r in rows
        ],
    }


@app.delete("/history/{pred_id}")
def delete_prediction(pred_id: str):
    """Hapus satu record prediksi berdasarkan ID."""
    conn = get_conn()

    existing = conn.execute(
        "SELECT id FROM predictions WHERE id = ?", (pred_id,)
    ).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(status_code=404, detail="Riwayat tidak ditemukan")

    conn.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
    conn.commit()
    conn.close()

    return {"message": "Riwayat berhasil dihapus", "id": pred_id}


@app.delete("/history/all/{user_id}")
def delete_all_history(user_id: str):
    """Hapus semua riwayat prediksi milik user."""
    conn = get_conn()

    user = conn.execute(
        "SELECT id FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="User tidak ditemukan")

    result = conn.execute(
        "DELETE FROM predictions WHERE user_id = ?", (user_id,)
    )
    deleted_count = result.rowcount
    conn.commit()
    conn.close()

    return {
        "message": f"{deleted_count} riwayat berhasil dihapus",
        "deleted_count": deleted_count,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STATS  (bonus — ringkasan statistik user)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/stats/{user_id}")
def get_stats(user_id: str):
    """
    Ringkasan statistik pemeriksaan milik user:
    total pemeriksaan, jumlah +TB, jumlah -TB, rata-rata confidence.
    """
    conn = get_conn()

    user = conn.execute(
        "SELECT id, nama FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="User tidak ditemukan")

    row = conn.execute(
        """
        SELECT
            COUNT(*)                                          AS total,
            SUM(CASE WHEN label LIKE '%+%' THEN 1 ELSE 0 END) AS tbc_positive,
            SUM(CASE WHEN label NOT LIKE '%+%' THEN 1 ELSE 0 END) AS tbc_negative,
            ROUND(AVG(confidence), 2)                         AS avg_confidence,
            MAX(timestamp)                                    AS last_check
        FROM predictions
        WHERE user_id = ?
        """,
        (user_id,),
    ).fetchone()
    conn.close()

    return {
        "user_id":        user_id,
        "nama":           user["nama"],
        "total":          row["total"] or 0,
        "tbc_positive":   row["tbc_positive"] or 0,
        "tbc_negative":   row["tbc_negative"] or 0,
        "avg_confidence": row["avg_confidence"] or 0.0,
        "last_check":     row["last_check"],
    }