"""Fungsi bersama untuk API dan monitoring: memuat bundle, preprocessing, prediksi, logging."""
import json
from pathlib import Path

import joblib
import pandas as pd

BUNDLE_PATH = "model_bundle.joblib"


def load_bundle(path=BUNDLE_PATH):
    # Peringatan: joblib/pickle bisa mengeksekusi kode saat dimuat.
    # Hanya muat file model yang kamu buat sendiri atau dari sumber tepercaya.
    return joblib.load(path)


def preprocess(records, bundle):
    """List of dict (data mentah) -> array numerik siap masuk model."""
    df = pd.DataFrame(records)
    for col, enc in bundle["encoders"].items():
        unknown = set(df[col].dropna().unique()) - set(enc.classes_)
        if unknown or df[col].isna().any():
            raise ValueError(
                f"Nilai '{col}' tidak dikenal: {sorted(map(str, unknown)) or 'kosong'}. "
                f"Nilai yang valid: {list(enc.classes_)}"
            )
        df[col] = enc.transform(df[col])
    df = df.reindex(columns=bundle["feature_names"])
    return bundle["imputer"].transform(df.to_numpy(dtype=float))


def predict(records, bundle):
    return bundle["model"].predict(preprocess(records, bundle))


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False,
                           default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
