from fastapi import FastAPI, Request
import sys
from datetime import datetime

app = FastAPI()

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
        closest_idx = -1
        closest_days = float('inf')
        for idx, prev in enumerate(prevs):
            if prev.get("study_description", "") == current_desc:
                try:
                    days = abs((datetime.strptime(current_date, "%Y-%m-%d") - datetime.strptime(prev.get("study_date", ""), "%Y-%m-%d")).days)
                    if days < closest_days:
                        closest_days = days
                        closest_idx = idx
                except Exception:
                    continue
        for i, prev in enumerate(prevs):
            study_id = prev.get("study_id", "")
            predictions.append({
                "case_id": case_id,
                "study_id": study_id,
                "predicted_is_relevant": i == closest_idx
            })
    return {"predictions": predictions}