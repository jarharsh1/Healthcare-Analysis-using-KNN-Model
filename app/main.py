"""FastAPI serving layer for the vertebral-column condition model.

Endpoints:
  GET  /health   - liveness/readiness, reports whether the model is loaded
  POST /predict  - classify a single patient's measurements
  POST /train    - re-run the full training pipeline (ops convenience)
"""

from fastapi import FastAPI, HTTPException

from app.schemas import PatientMeasurements, PredictionResponse
from vertebral.logger import logger

app = FastAPI(
    title="Vertebral Column Condition API",
    version="0.1.0",
    description="Predicts a patient's condition (Normal / Type_H / Type_S) "
    "from six biomechanics measurements.",
)

_pipeline = None


def _get_pipeline():
    """Lazy-load the model so the app can boot even before training runs."""
    global _pipeline
    if _pipeline is None:
        from vertebral.pipeline.prediction import PredictionPipeline

        _pipeline = PredictionPipeline()
    return _pipeline


@app.get("/health")
def health() -> dict:
    model_ready = True
    try:
        _get_pipeline()
    except Exception:  # model artifact not present yet
        model_ready = False
    return {"status": "ok", "model_loaded": model_ready}


@app.post("/predict", response_model=PredictionResponse)
def predict(measurements: PatientMeasurements) -> PredictionResponse:
    try:
        pipeline = _get_pipeline()
    except Exception as exc:
        logger.error("Model not available: %s", exc)
        raise HTTPException(status_code=503, detail="Model not trained yet.") from exc

    result = pipeline.predict(measurements.model_dump())
    return PredictionResponse(**result)


@app.post("/train")
def train() -> dict:
    """Trigger a full retrain. Clears cached model/encoder on success."""
    global _pipeline
    try:
        from main import main as run_pipeline
        from vertebral.pipeline.prediction import _load_label_encoder, _load_model

        run_pipeline()
        # Drop caches so the next request uses the freshly trained artifacts.
        _load_model.cache_clear()
        _load_label_encoder.cache_clear()
        _pipeline = None
        return {"status": "retrained"}
    except Exception as exc:
        logger.error("Retrain failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
