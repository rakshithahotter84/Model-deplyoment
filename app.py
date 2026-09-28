from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

MODEL_PATH = "model.pkl"

app = FastAPI(
    title="ML Model Deployment API",
    description="Simple FastAPI service for machine-learning inference.",
    version="1.0.0",
)

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    model = None
    model_load_error = str(exc)


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Four numeric features expected by the demo Iris model.",
        examples=[[5.1, 3.5, 1.4, 0.2]],
    )


@app.get("/")
def root():
    return {
        "message": "ML Model API is running",
        "docs": "/docs",
        "endpoint": "/predict",
    }


@app.get("/health")
def health():
    if model is None:
        return {"status": "error", "model_loaded": False, "error": model_load_error}
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model could not be loaded.")

    try:
        features = np.array(request.features, dtype=float).reshape(1, -1)
        prediction = model.predict(features)
        result = {"prediction": int(prediction[0])}

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            result["probabilities"] = [round(float(x), 6) for x in probabilities]

        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")
