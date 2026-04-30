from fastapi import FastAPI, Request
import sys
from datetime import datetime
import joblib
import numpy as np
import os

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "relevance_model.joblib")
clf = joblib.load(MODEL_PATH)

def extract_modality(desc):
    desc = desc.lower()
    for mod in ['ct', 'mri', 'us', 'xray', 'echo', 'pet']:
        if mod in desc:
            return mod
    return 'other'

def time_bucket(days):
    if days < 0:
        return -1
    if days <= 7:
        return 0
    if days <= 30:
        return 1
    if days <= 180:
        return 2
    if days <= 365:
        return 3
    return 4

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    predictions = []
    for case in data.get("cases", []):
        case_id = case.get("case_id", "")
        current = case.get("current_study", {})
        current_desc = current.get("study_description", "")
        current_date = current.get("study_date", "")
        curr_dt = None
        try:
            curr_dt = datetime.strptime(current_date, "%Y-%m-%d")
        except Exception:
            pass
        current_mod = extract_modality(current_desc)
        for prev in case.get("prior_studies", []):
            study_id = prev.get("study_id", "")
            prev_desc = prev.get("study_description", "")
            prev_date = prev.get("study_date", "")
            prev_dt = None
            try:
                prev_dt = datetime.strptime(prev_date, "%Y-%m-%d")
            except Exception:
                pass
            same_desc = int(prev_desc == current_desc)
            days_diff = (curr_dt - prev_dt).days if curr_dt and prev_dt else 9999
            keyword_overlap = len(set(current_desc.lower().split()) & set(prev_desc.lower().split()))
            fuzzy1 = int(current_desc.lower() in prev_desc.lower())
            fuzzy2 = int(prev_desc.lower() in current_desc.lower())
            prev_mod = extract_modality(prev_desc)
            same_modality = int(prev_mod == current_mod)
            days_bucket = time_bucket(days_diff)
            same_patient = int(case.get('patient_id', '') != '' and case.get('patient_id', '') == case.get('patient_id', ''))
            X = np.array([[same_desc, days_diff, keyword_overlap, fuzzy1, fuzzy2, same_modality, days_bucket, same_patient]])
            pred = clf.predict(X)[0]
            predictions.append({
                "case_id": case_id,
                "study_id": study_id,
                "predicted_is_relevant": bool(pred)
            })
    return {"predictions": predictions}