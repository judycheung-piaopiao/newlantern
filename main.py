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
        prevs = case.get("prior_studies", [])
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