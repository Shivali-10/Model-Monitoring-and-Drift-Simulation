from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from typing import List

os.chdir(r"C:\Users\HP\Desktop\fraud_drift_project")

app = FastAPI(
    title="🛡️ Adaptive Drift Monitor API",
    description="ML Model Monitoring & Drift Detection REST API",
    version="2.0"
)

# Load model and data on startup
model  = joblib.load("fraud_model.pkl")
week1  = pd.read_csv("week1_baseline.csv")
target = "Class"
features = [col for col in week1.columns if col != target]

# ── Request Models ────────────────────────────────────────────
class Transaction(BaseModel):
    features: List[float]

class DriftRequest(BaseModel):
    reference_size: int = 1000
    current_size:   int = 1000
    noise_level:    float = 3.0

# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "Adaptive Drift Monitor API",
        "version": "2.0",
        "status": "running",
        "endpoints": ["/health", "/predict", "/detect-drift", "/retrain"]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Random Forest Classifier",
        "features": len(features),
        "baseline_rows": len(week1)
    }

@app.post("/predict")
def predict(transaction: Transaction):
    if len(transaction.features) != len(features):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(features)} features, got {len(transaction.features)}"
        )
    X = np.array(transaction.features).reshape(1, -1)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    return {
        "prediction": "FRAUD" if prediction == 1 else "NORMAL",
        "fraud_probability": round(float(probability[1]), 4),
        "normal_probability": round(float(probability[0]), 4),
        "confidence": round(float(max(probability)), 4)
    }

@app.post("/detect-drift")
def detect_drift(request: DriftRequest):
    reference = week1[features].head(request.reference_size)
    current   = week1[features].head(request.current_size).copy()

    for col in features[:3]:
        current[col] = current[col] + np.random.normal(
            request.noise_level, 2.0, len(current)
        )

    drift_results = []
    for feat in features[:10]:
        ks_stat, p_val = stats.ks_2samp(reference[feat], current[feat])
        drift_results.append({
            "feature":      feat,
            "ks_statistic": round(float(ks_stat), 4),
            "p_value":      round(float(p_val), 6),
            "drift":        bool(ks_stat > 0.1)
        })

    drifted    = sum(1 for r in drift_results if r["drift"])
    avg_ks     = round(np.mean([r["ks_statistic"] for r in drift_results]), 4)
    drift_detected = drifted > 0

    return {
        "drift_detected":  drift_detected,
        "drifted_features": drifted,
        "total_features":   len(drift_results),
        "avg_ks_score":     avg_ks,
        "noise_level":      request.noise_level,
        "feature_results":  drift_results,
        "recommendation":   "RETRAIN" if drift_detected else "STABLE"
    }

@app.post("/retrain")
def retrain():
    X = week1[features]
    y = week1[target]
    try:
        sm = SMOTE(random_state=42)
        Xr, yr = sm.fit_resample(X, y)
    except:
        Xr, yr = X, y

    ne