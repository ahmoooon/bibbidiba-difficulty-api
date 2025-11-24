from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
import json

from google.cloud import firestore
from google.oauth2 import service_account

app = FastAPI(title="Bibbi Difficulty Service")

# ---- Load model ----
model = joblib.load("difficulty_model.pkl")

# ---- Load feature metadata (flexible: list OR dict) ----
_feature_meta = joblib.load("difficulty_model_features.pkl")
if isinstance(_feature_meta, dict):
    feature_cols = _feature_meta.get("feature_names", [])
else:
    # old style: plain list
    feature_cols = _feature_meta

if not feature_cols:
    raise RuntimeError("feature_cols is empty – check difficulty_model_features.pkl")

# Read JSON from environment variable
service_account_info = json.loads(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
creds = service_account.Credentials.from_service_account_info(service_account_info)

# Firebase project ID
PROJECT_ID = "bibbidiba"

# Create Firestore client with explicit credentials + project ID
db = firestore.Client(project=PROJECT_ID, credentials=creds)


# ---------- REQUEST / RESPONSE SCHEMAS ----------

class DifficultyRequest(BaseModel):
    userId: str


class DifficultyResponse(BaseModel):
    difficulty_action: str          # "Decrease" | "Maintain" | "Increase"
    confidence: float               # probability of chosen class
    probabilities: dict             # { "Decrease": 0.12, "Maintain": 0.5, ... }


# ---------- FEATURE ENGINEERING HELPERS ----------

PASS_THRESHOLD = 0.5


def compute_streaks(outcomes):
    """
    outcomes: list of 0/1 (fail/success).
    Returns: (max_success_streak, max_fail_streak)
    """
    max_s = max_f = 0
    cur_s = cur_f = 0
    for o in outcomes:
        if o == 1:
            cur_s += 1
            cur_f = 0
        else:
            cur_f += 1
            cur_s = 0
        max_s = max(max_s, cur_s)
        max_f = max(max_f, cur_f)
    return max_s, max_f


def build_words_meta(db):
    """
    Load all words once, build a mapping:
      wordId -> {chapterId, lengthClass, category}
    category is derived from chapterId, e.g. "ch_alphabet_1" -> "ch_alphabet".
    """
    chapters_ref = db.collection("chapters")
    words_meta = {}

    for ch_doc in chapters_ref.stream():
        ch_id = ch_doc.id
        words_ref = chapters_ref.document(ch_id).collection("words")
        for w_doc in words_ref.stream():
            w_data = w_doc.to_dict() or {}
            word_id = w_doc.id
            chapter_id = w_data.get("chapterId", ch_id)
            length_class = w_data.get("lengthClass")

            # derive category from chapter id, e.g. "ch_alphabet_1" -> "ch_alphabet"
            if isinstance(chapter_id, str) and "_" in chapter_id:
                category = "_".join(chapter_id.split("_")[:-1])
            else:
                category = chapter_id

            words_meta[word_id] = {
                "chapterId": chapter_id,
                "lengthClass": length_class,
                "category": category,
            }

    return words_meta


def build_features_for_user(user_id: str) -> pd.DataFrame:
    """
    Query Firestore for this child (userId), compute one row of features
    per finished gameSession. Returns a DataFrame like your notebook's
    features_df (including catAcc_* and lenAcc_* columns).
    """
    # ---------------------------
    # 1) Load sessions for child
    # ---------------------------
    sessions_ref = (
        db.collection("users")
          .document(user_id)
          .collection("gameSessions")
    )

    sessions = []
    # only finished sessions: endTime != None
    for doc in sessions_ref.where("endTime", "!=", None).stream():
        data = doc.to_dict() or {}
        data.setdefault("gSessionId", doc.id)
        sessions.append(data)

    if not sessions:
        raise HTTPException(
            status_code=404,
            detail="No finished gameSessions found for this user."
        )

    sessions_df = pd.DataFrame(sessions)

    # Convert Firestore Timestamps to pandas datetime
    for col in ["startTime", "endTime"]:
        if col in sessions_df.columns:
            sessions_df[col] = pd.to_datetime(sessions_df[col])

    # ---------------------------
    # 2) Load attempts for child
    # ---------------------------
    attempts_ref = db.collection_group("wordAttempts").where("userId", "==", user_id)
    attempts = []
    for doc in attempts_ref.stream():
        row = doc.to_dict() or {}
        row.setdefault("attemptId", doc.id)
        attempts.append(row)

    if not attempts:
        raise HTTPException(
            status_code=404,
            detail="No wordAttempts found for this user."
        )

    attempts_df = pd.DataFrame(attempts)

    # Fix dtypes
    attempts_df["accuracy"] = attempts_df["accuracy"].astype(float)
    attempts_df["timeTakenSec"] = attempts_df["timeTakenSec"].astype(float)
    attempts_df["createdAt"] = pd.to_datetime(attempts_df["createdAt"])

    # ---------------------------
    # 3) Attach category & lengthClass via words_meta
    # ---------------------------
    words_meta = build_words_meta(db)

    def map_category(word_id):
        info = words_meta.get(word_id)
        return info["category"] if info else None

    def map_length(word_id):
        info = words_meta.get(word_id)
        return info["lengthClass"] if info else None

    attempts_df["category"] = attempts_df["wordId"].map(map_category)
    attempts_df["lengthClass"] = attempts_df["wordId"].map(map_length)

    # TOTAL_CATEGORIES = number of distinct categories in catalog
    all_cats = {v["category"] for v in words_meta.values() if v.get("category")}
    TOTAL_CATEGORIES = len(all_cats) if all_cats else 1  # avoid divide-by-zero

    # ---------------------------
    # 4) Per-session base features (1,2,3,4,5,6,7,11,12,13,15)
    # ---------------------------
    features_rows = []
    attempts_by_session = dict(tuple(attempts_df.groupby("gSessionId")))

    for _, sess in sessions_df.iterrows():
        sid = sess.get("gSessionId")
        game_type = sess.get("gameType")
        duration_min = float(sess.get("durationMin", 0.0) or 0.0)
        avg_acc = float(sess.get("avgAccuracy", 0.0) or 0.0)  # 0..1
        target_count = int(sess.get("targetCount", 0) or 0)
        completed_count = int(sess.get("completedCount", 0) or 0)
        success_count = int(sess.get("successCount", 0) or 0)

        sess_attempts = attempts_by_session.get(sid)
        if sess_attempts is None or sess_attempts.empty:
            # no attempts in this session
            continue

        # sort by time
        sess_attempts = sess_attempts.sort_values("createdAt")

        total_attempts = len(sess_attempts)
        unique_words = sess_attempts["wordId"].nunique()
        attempts_per_word = total_attempts / unique_words if unique_words > 0 else 0.0

        mean_time = float(sess_attempts["timeTakenSec"].mean() or 0.0)
        std_time = float(sess_attempts["timeTakenSec"].std(ddof=0) or 0.0)

        # 0/1 outcomes using PASS_THRESHOLD (Feature 4 + 5)
        outcome_seq = (sess_attempts["accuracy"] >= PASS_THRESHOLD).astype(int).tolist()
        max_s_streak, max_f_streak = compute_streaks(outcome_seq)

        # successRate
        if completed_count > 0:
            success_rate = success_count / completed_count
        else:
            success_rate = float(np.mean(outcome_seq)) if outcome_seq else 0.0

        # session duration in seconds
        session_duration_sec = max(duration_min * 60.0, 1e-6)

        # active time = sum of speaking time
        active_time = float(sess_attempts["timeTakenSec"].sum() or 0.0)
        active_ratio = active_time / session_duration_sec

        # abandonRate
        if target_count > 0:
            completion_ratio = completed_count / target_count
            completion_ratio = max(0.0, min(1.0, completion_ratio))
        else:
            completion_ratio = 1.0
        abandon_rate = 1.0 - completion_ratio

        # Feature 6: engagementIndex
        engagement_index = (
            0.6 * active_ratio +
            0.3 * success_rate +
            0.1 * (1.0 - abandon_rate)
        )

        # Feature 12: totalWordsAttempted
        total_words_attempted = unique_words

        # Feature 13: categoryVariety
        categories_in_session = set(sess_attempts["category"].dropna().tolist())
        num_cats = len(categories_in_session)
        category_variety = num_cats / TOTAL_CATEGORIES if TOTAL_CATEGORIES > 0 else 0.0

        # Feature 15: attentionStabilityIndex
        if mean_time > 0.0:
            attention_stability = 1.0 - (std_time / mean_time)
            attention_stability = max(0.0, min(1.0, attention_stability))
        else:
            attention_stability = 0.0

        # Feature 11: preferredTimeOfDay
        start_time = sess.get("startTime")
        if pd.isna(start_time):
            time_of_day = 2  # default: evening/night
        else:
            hour = start_time.hour
            if 6 <= hour <= 11:
                time_of_day = 0  # morning
            elif 12 <= hour <= 17:
                time_of_day = 1  # afternoon
            else:
                time_of_day = 2  # evening/night

        features_rows.append({
            "userId": user_id,
            "gSessionId": sid,
            "gameType": game_type,
            "startTime": start_time,
            "endTime": sess.get("endTime"),

            # Feature 1
            "avgAccuracy": avg_acc,
            "avgAccuracyPct": avg_acc * 100.0,

            # Feature 2
            "attemptsPerWord": attempts_per_word,

            # Feature 3 & 15
            "meanTimeTakenSec": mean_time,
            "stdTimeTakenSec": std_time,
            "attentionStabilityIndex": attention_stability,

            # Feature 4
            "successRate": success_rate,

            # Feature 5
            "maxSuccessStreak": max_s_streak,
            "maxFailStreak": max_f_streak,

            # Feature 6
            "engagementIndex": engagement_index,
            "activeTimeSec": active_time,
            "sessionDurationMin": duration_min,
            "abandonRate": abandon_rate,

            # Feature 7
            "durationMin": duration_min,

            # Feature 11
            "preferredTimeOfDay": time_of_day,

            # Feature 12
            "totalWordsAttempted": total_words_attempted,

            # Feature 13
            "categoryVariety": category_variety,

            # Feature 8 & 14 placeholders, filled below
            "retentionScore": np.nan,
            "improvementTrend": np.nan,
        })

    if not features_rows:
        raise HTTPException(
            status_code=404,
            detail="No usable sessions with attempts for this user."
        )

    features_df = pd.DataFrame(features_rows).sort_values("startTime").reset_index(drop=True)

    # ---------------------------
    # 5) Retention + Improvement
    # ---------------------------

    # Retention Score: difference vs previous session's avgAccuracy
    features_df["retentionScore"] = features_df["avgAccuracy"].diff()

    # Improvement Trend: vs mean of previous 3 sessions
    features_df["avgAccuracyPrev3"] = (
        features_df["avgAccuracy"]
        .rolling(window=3, min_periods=1)
        .mean()
        .shift(1)
    )
    features_df["improvementTrend"] = (
        features_df["avgAccuracy"] - features_df["avgAccuracyPrev3"]
    )
    
    features_df["retentionScore"].fillna(0.0, inplace=True)
    features_df["improvementTrend"].fillna(0.0, inplace=True)

    return features_df


# ---------- ROUTES ----------

@app.get("/")
def root():
    return {"status": "ok", "message": "Bibbi Difficulty Service running"}


@app.post("/predict", response_model=DifficultyResponse)
def predict(req: DifficultyRequest):
    # 1) Build features for this child from Firestore
    try:
        features_df = build_features_for_user(req.userId)
    except HTTPException:
        # re-raise clean API errors
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing features: {e}")

    if features_df is None or features_df.empty:
        raise HTTPException(status_code=404, detail="No finished sessions for this user")

    # 2) Pick the most recent session as the “current state”
    row = features_df.sort_values("startTime").iloc[-1]

    print("=== Latest Session Features for Debug ===")
    print(row.to_dict())  # will appear in Render logs

    # Optionally: export to /tmp/features_debug.csv (safe writable path)
    features_df.tail(1).to_csv("/tmp/features_debug.csv", index=False)
    print("Features saved to /tmp/features_debug.csv")

    # 3) Build input vector in EXACT training order
    try:
        x_vals = []
        int_cols = ["preferredTimeOfDay", "totalWordsAttempted"]
        for col in feature_cols:
            val = row.get(col, 0)  # default 0 if missing
            if pd.isna(val):
                val = 0.0            # replace NaN with 0.0
            if col in int_cols:
                val = int(val)
            x_vals.append(val)
        
        x = np.array([x_vals], dtype=float)
    except KeyError as e:
        missing = str(e)
        raise HTTPException(
            status_code=500,
            detail=(
                f"Feature column {missing} missing in features_df. "
                f"Check difficulty_model_features.pkl vs computation."
            ),
        )

    # 4) Run model
    preds = model.predict(x)
    probs = model.predict_proba(x)[0]
    classes = model.classes_

    class_probs = {str(cls): float(p) for cls, p in zip(classes, probs)}
    chosen = str(preds[0])
    confidence = float(max(probs))

    return DifficultyResponse(
        difficulty_action=chosen,
        confidence=confidence,
        probabilities=class_probs,
    )
