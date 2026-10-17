"""API prediksi harga rumah (Flask).
Dev      : python app.py
Produksi : gunicorn -w 1 --threads 4 -b 0.0.0.0:8000 app:app
"""
import math
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request

from ml_utils import append_jsonl, load_bundle, predict

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
PRED_LOG = LOG_DIR / "predictions.jsonl"
FEEDBACK_LOG = LOG_DIR / "feedback.jsonl"
MAX_ROWS = 1000

bundle = load_bundle()
app = Flask(__name__)

lock = threading.Lock()
stats = {"requests": 0, "errors": 0}
latencies = deque(maxlen=1000)

# nama fitur: (nilai minimum, nilai maksimum, boleh kosong). Sesuaikan jika kolom dataset berbeda.
FIELDS = {
    "longitude": (-125, -114, False),
    "latitude": (32, 43, False),
    "housing_median_age": (0, 100, False),
    "total_rooms": (1, None, False),
    "total_bedrooms": (0, None, True),    # boleh kosong -> diimputasi median training
    "population": (1, None, False),
    "households": (1, None, False),
    "median_income": (0.01, None, False),
}
CATEGORY = "ocean_proximity"
VALID_CATEGORIES = list(bundle["encoders"][CATEGORY].classes_)


def validate(record):
    clean, errors = {}, []
    for name, (lo, hi, optional) in FIELDS.items():
        v = record.get(name)
        if v is None:
            if optional:
                clean[name] = None
            else:
                errors.append(f"'{name}' wajib diisi")
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            errors.append(f"'{name}' harus berupa angka")
        elif (lo is not None and v < lo) or (hi is not None and v > hi):
            errors.append(f"'{name}' di luar rentang [{lo}, {hi}]")
        else:
            clean[name] = float(v)
    cat = record.get(CATEGORY)
    if cat not in VALID_CATEGORIES:
        errors.append(f"'{CATEGORY}' harus salah satu dari {VALID_CATEGORIES}")
    clean[CATEGORY] = cat
    return clean, errors


def count(error=False, latency_ms=None):
    with lock:
        stats["requests"] += 1
        stats["errors"] += int(error)
        if latency_ms is not None:
            latencies.append(latency_ms)


@app.get("/health")
def health():
    return jsonify(status="ok", model_version=bundle["model_version"])


@app.post("/predict")
def predict_endpoint():
    """Body: {"data": [ {fitur...}, ... ]}  atau satu objek fitur langsung."""
    start = time.perf_counter()
    payload = request.get_json(silent=True)
    if payload is None:
        count(error=True)
        return jsonify(error="Body harus berupa JSON"), 400

    items = payload.get("data", payload) if isinstance(payload, dict) else payload
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list) or not items or not all(isinstance(i, dict) for i in items):
        count(error=True)
        return jsonify(error="'data' harus berisi satu atau lebih objek fitur"), 400
    if len(items) > MAX_ROWS:
        count(error=True)
        return jsonify(error=f"Maksimal {MAX_ROWS} baris per request"), 400

    cleaned, problems = [], []
    for idx, rec in enumerate(items):
        clean, errs = validate(rec)
        cleaned.append(clean)
        if errs:
            problems.append({"index": idx, "errors": errs})
    if problems:
        count(error=True)
        return jsonify(error="Input tidak valid", detail=problems), 422

    try:
        preds = predict(cleaned, bundle)
    except Exception:
        app.logger.exception("Prediksi gagal")
        count(error=True)
        return jsonify(error="Prediksi gagal"), 500

    latency_ms = (time.perf_counter() - start) * 1000
    count(latency_ms=latency_ms)
    now = datetime.now(timezone.utc).isoformat()
    ids = []
    for rec, pred in zip(cleaned, preds):
        rid = uuid.uuid4().hex
        ids.append(rid)
        append_jsonl(PRED_LOG, {
            "request_id": rid, "timestamp": now, "input": rec, "prediction": float(pred),
            "latency_ms": round(latency_ms, 2), "model_version": bundle["model_version"],
        })
    return jsonify(prediction=[round(float(p), 2) for p in preds], request_ids=ids,
                   model_version=bundle["model_version"])


@app.post("/feedback")
def feedback():
    """Kirim nilai aktual setelah diketahui: {"request_id": "...", "actual_value": 452600}"""
    body = request.get_json(silent=True) or {}
    rid, actual = body.get("request_id"), body.get("actual_value")
    if (not isinstance(rid, str) or isinstance(actual, bool)
            or not isinstance(actual, (int, float)) or not math.isfinite(actual) or actual <= 0):
        return jsonify(error="Wajib: request_id (teks) dan actual_value (angka > 0)"), 422
    append_jsonl(FEEDBACK_LOG, {"request_id": rid, "actual_value": float(actual),
                                "timestamp": datetime.now(timezone.utc).isoformat()})
    return jsonify(status="tersimpan")


@app.get("/metrics")
def metrics():
    with lock:
        lat = np.array(latencies) if latencies else np.array([0.0])
        n, err = stats["requests"], stats["errors"]
    return jsonify(requests=n, errors=err, error_rate=round(err / n, 4) if n else 0.0,
                   latency_ms_avg=round(float(lat.mean()), 2),
                   latency_ms_p95=round(float(np.percentile(lat, 95)), 2),
                   model_version=bundle["model_version"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
