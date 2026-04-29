from fastapi import FastAPI, Request
import sys

app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    print("Received data:", data, file=sys.stderr)
    predictions = []
    for case in data.get("cases", []):
        case_id = case.get("case_id", "")
        # 兼容字段名
        prevs = case.get("previous_examinations") or case.get("previous_studies") or []
        if not isinstance(prevs, list):
            prevs = []
        for i, prev in enumerate(prevs):
            study_id = prev.get("study_id", "")
            predictions.append({
                "case_id": case_id,
                "study_id": study_id,
                "predicted_is_relevant": i == 0
            })
    print(f"Predictions returned: {len(predictions)}", file=sys.stderr)
    return {"predictions": predictions}