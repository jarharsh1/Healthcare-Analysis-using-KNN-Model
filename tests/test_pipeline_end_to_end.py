"""End-to-end smoke test: run the whole pipeline, then classify via the API."""

import json

from fastapi.testclient import TestClient


def test_full_pipeline_and_api():
    from main import main as run_pipeline

    run_pipeline()

    with open("artifacts/model_evaluation/metrics.json") as f:
        metrics = json.load(f)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    # Three well-separated biomechanics classes; a real model clears this.
    assert metrics["accuracy"] > 0.6

    from app.main import app

    client = TestClient(app)
    assert client.get("/health").status_code == 200

    payload = {
        "P_incidence": 63.03,
        "P_tilt": 22.55,
        "L_angle": 39.61,
        "S_slope": 40.48,
        "P_radius": 98.67,
        "S_Degree": -0.25,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in {"Normal", "Type_H", "Type_S"}
    # Probabilities cover all three classes and sum to ~1.
    assert set(body["probabilities"]) == {"Normal", "Type_H", "Type_S"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6
