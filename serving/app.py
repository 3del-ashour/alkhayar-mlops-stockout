from __future__ import annotations

from fastapi import FastAPI, HTTPException

from serving.model_loader import load_model, prepare_features
from serving.schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="Stockout Prediction API")


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    try:
        model, version = load_model()
        features = prepare_features(req.dict())
        prob = float(model.predict(features)[0])
        prediction = int(prob >= 0.5)
        return PredictionResponse(prediction=prediction, probability=prob, model_version=str(version))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
