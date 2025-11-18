# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Bibbi Difficulty Service")

# ---- Load model & feature list ----
model = joblib.load("difficulty_model.pkl")

try:
    feature_info = joblib.load("difficulty_model_features.pkl")
    FEATURE_NAMES = feature_info.get("feature_names", [])
except Exception:
    # fallback: hardcode or leave empty
    FEATURE_NAMES = [
        "avgAccuracy",
        "attemptsPerWord",
        "meanTimeTakenSec",
        "attentionStabilityIndex",
        "successRate",
        "maxSuccessStreak",
        "maxFailStreak",
        "engagementIndex",
        "sessionDurationMin",
        "preferredTimeOfDay",
        "totalWordsAttempted",
        "categoryVariety",
        "retentionScore",
        "improvementTrend",
    ]

# ---- Request + Response schemas ----

class DifficultyRequest(BaseModel):
    # must match your features_df columns you pass from Flutter
    avgAccuracy: float
    attemptsPerWord: float
    meanTimeTakenSec: float
    attentionStabilityIndex: float
    successRate: float
    maxSuccessStreak: int
    maxFailStreak: int
    engagementIndex: float
    sessionDurationMin: float
    preferredTimeOfDay: int  # 0=morning,1=afternoon,2=evening
    totalWordsAttempted: int
    categoryVariety: float
    retentionScore: float
    improvementTrend: float


class DifficultyResponse(BaseModel):
    difficulty_action: str          # "Decrease" | "Maintain" | "Increase"
    confidence: float               # model probability for chosen class
    probabilities: dict             # dict[class] = prob


# ---- Helper to prepare the feature vector ----

def request_to_vector(req: DifficultyRequest):
    # Create vector in same order as training
    x = np.array([
        req.avgAccuracy,
        req.attemptsPerWord,
        req.meanTimeTakenSec,
        req.attentionStabilityIndex,
        req.successRate,
        req.maxSuccessStreak,
        req.maxFailStreak,
        req.engagementIndex,
        req.sessionDurationMin,
        req.preferredTimeOfDay,
        req.totalWordsAttempted,
        req.categoryVariety,
        req.retentionScore,
        req.improvementTrend,
    ], dtype=float).reshape(1, -1)
    return x

# ---- Routes ----

@app.get("/")
def root():
    return {"status": "ok", "message": "Bibbi Difficulty Service running"}

@app.post("/predict", response_model=DifficultyResponse)
def predict(req: DifficultyRequest):
    x = request_to_vector(req)
    preds = model.predict(x)
    probs = model.predict_proba(x)[0]

    classes = model.classes_
    class_probs = {str(cls): float(p) for cls, p in zip(classes, probs)}
    chosen = preds[0]
    confidence = float(max(probs))

    return DifficultyResponse(
        difficulty_action=str(chosen),
        confidence=confidence,
        probabilities=class_probs,
    )