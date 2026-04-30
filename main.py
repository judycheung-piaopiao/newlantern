from fastapi import FastAPI, Request
import sys
from datetime import datetime
import joblib
import numpy as np
import os

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "relevance_model.joblib")
clf = joblib.load(MODEL_PATH)

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    predictions = []
    for case in data.get("cases", []):
        case_id = case.get("case_id", "")
        current = case.get("current_study", {})
        current_desc = current.get("study_description", "")
        current_date = current.get("study_date", "")
        prevs = case.get("prior_studies", [])
        for prev in prevs:
            study_id = prev.get("study_id", "")
            prev_desc = prev.get("study_description", "")
            prev_date = prev.get("study_date", "")
            try:
                prev_dt = datetime.strptime(prev_date, "%Y-%m-%d")
            except Exception:
                prev_dt = None
            same_desc = int(prev_desc == current_desc)
            curr_dt = datetime.strptime(current_date, "%Y-%m-%d")
            days_diff = (curr_dt - prev_dt).days if curr_dt and prev_dt else 9999
            keyword_overlap = len(set(current_desc.split()) & set(prev_desc.split()))
            X = np.array([[same_desc, days_diff, keyword_overlap]])
            pred = clf.predict(X)[0]
            predictions.append({
                "case_id": case_id,
                "study_id": study_id,
                "predicted_is_relevant": bool(pred)
            })
    return {"predictions": predictions}