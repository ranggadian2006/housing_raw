"""Monitoring: kesehatan layanan, data drift, dan performa model. Bisa dijalankan terjadwal: python monitoring.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, r2_score

from ml_utils import load_bundle, preprocess

THRESHOLDS = {
    "psi_warn": 0.10,
    "psi_alert": 0.25,
    "mae_degradation": 0.20,   # MAE boleh naik maksimal 20% dari baseline
    "p95_latency_ms": 500,
    "min_samples": 100,        # di bawah ini hasil drift tidak andal
}


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    with path.open(encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


def psi(expected, actual):
    """Population Stability Index. <0.1 stabil, 0.1-0.25 waspada, >0.25 drift."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(np.unique(expected)) <= 10:          # fitur diskrit/kategori -> bandingkan proporsi
        cats = np.union1d(np.unique(expected), np.unique(actual))
        e = np.array([(expected == c).mean() for c in cats])
        a = np.array([(actual == c).mean() for c in cats])
    else:                                       # fitur kontinu -> bin berdasarkan kuantil data training
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, 11)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference, live, features):
    rows = []
    for col in features:
        p = psi(reference[col], live[col])
        ks = ks_2samp(reference[col], live[col])
        status = ("DRIFT" if p >= THRESHOLDS["psi_alert"]
                  else "WASPADA" if p >= THRESHOLDS["psi_warn"] else "STABIL")
        rows.append({"fitur": col, "psi": round(p, 3), "ks_stat": round(float(ks.statistic), 3),
                     "ks_pvalue": round(float(ks.pvalue), 4), "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def run_monitoring(log_dir="logs", bundle_path="model_bundle.joblib",
                   reference_path="reference_data.csv", save=True, verbose=True):
    bundle = load_bundle(bundle_path)
    features = bundle["feature_names"]
    preds = read_jsonl(Path(log_dir) / "predictions.jsonl")
    fb = read_jsonl(Path(log_dir) / "feedback.jsonl")
    alerts, result = [], {"n_predictions": len(preds), "n_feedback": len(fb)}

    if preds.empty:
        alerts.append("Belum ada log prediksi.")
        return {"alerts": alerts, **result}

    # 1) Layanan
    p95 = float(np.percentile(preds["latency_ms"], 95))
    result["latency_p95_ms"] = round(p95, 1)
    if p95 > THRESHOLDS["p95_latency_ms"]:
        alerts.append(f"Latensi p95 {p95:.0f} ms melebihi {THRESHOLDS['p95_latency_ms']} ms.")

    # 2) Data drift
    if len(preds) < THRESHOLDS["min_samples"]:
        alerts.append(f"Baru {len(preds)} prediksi (< {THRESHOLDS['min_samples']}): drift belum bisa dinilai andal.")
        drift = None
    else:
        reference = pd.read_csv(reference_path)
        live = pd.DataFrame(preprocess(list(preds["input"]), bundle), columns=features)
        drift = drift_report(reference, live, features)
        for _, r in drift[drift["status"] == "DRIFT"].iterrows():
            alerts.append(f"Drift pada '{r['fitur']}' (PSI={r['psi']}).")
    result["drift"] = drift

    # 3) Performa model (hanya untuk request yang sudah punya nilai aktual)
    perf = None
    if not fb.empty:
        m = preds.merge(fb[["request_id", "actual_value"]], on="request_id")
        if len(m) >= 30:
            mae = float(mean_absolute_error(m["actual_value"], m["prediction"]))
            r2 = float(r2_score(m["actual_value"], m["prediction"]))
            base = bundle["baseline"]
            perf = {"n": len(m), "MAE": round(mae, 1), "R2": round(r2, 3),
                    "MAE_baseline": round(base["MAE"], 1), "R2_baseline": round(base["R2"], 3)}
            if mae > base["MAE"] * (1 + THRESHOLDS["mae_degradation"]):
                alerts.append(f"MAE {mae:,.0f} naik {mae / base['MAE'] - 1:.0%} dari baseline {base['MAE']:,.0f}. "
                              "Pertimbangkan retraining.")
    result["performance"] = perf
    result["alerts"] = alerts

    if verbose:
        print(f"Prediksi: {result['n_predictions']} | umpan balik: {result['n_feedback']} | "
              f"latensi p95: {result['latency_p95_ms']} ms")
        if drift is not None:
            print("\nDrift fitur (PSI tertinggi di atas):")
            print(drift.to_string(index=False))
        if perf:
            print("\nPerforma:", perf)
        print("\nALERT:" if alerts else "\nTidak ada alert.")
        for a in alerts:
            print(" -", a)

    if save:
        out = {k: v for k, v in result.items() if k != "drift"}
        out["drift"] = None if drift is None else drift.to_dict("records")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        (Path(log_dir) / "monitoring_report.json").write_text(json.dumps(out, indent=2, default=str))
    return result


if __name__ == "__main__":
    run_monitoring()
