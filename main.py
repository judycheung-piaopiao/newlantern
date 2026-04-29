from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI()

class Prediction(BaseModel):
    case_id: str
    study_id: str
    predicted_is_relevant: bool

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    predictions = []
    for case in data.get("cases", []):
        case_id = case["case_id"]
        prevs = case.get("previous_examinations", [])
        for i, prev in enumerate(prevs):
            predictions.append({
                "case_id": case_id,
                "study_id": prev["study_id"],
                "predicted_is_relevant": i == 0
            })
    return {"predictions": predictions}
