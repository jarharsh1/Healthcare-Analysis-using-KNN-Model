"""Unit tests for the prediction pipeline.

The missing-feature guard runs without any trained model (we bypass __init__).
The probability-labeling test builds a tiny in-memory model + encoder, so it uses
sklearn but not the full pipeline / disk artifacts.
"""

import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from vertebral.pipeline.prediction import PredictionPipeline

FEATURES = ["P_incidence", "P_tilt", "L_angle", "S_slope", "P_radius", "S_Degree"]


def _bare_pipeline():
    p = PredictionPipeline.__new__(PredictionPipeline)
    p.feature_columns = tuple(FEATURES)
    p.model = None
    p.label_encoder = None
    return p


def test_missing_feature_raises_clear_error():
    p = _bare_pipeline()
    partial = {c: 1.0 for c in FEATURES[:-1]}  # drop S_Degree
    with pytest.raises(ValueError, match="Missing required feature field"):
        p.predict(partial)
    with pytest.raises(ValueError, match="S_Degree"):
        p.predict(partial)


def test_probabilities_labeled_by_model_class_order():
    # Encoder knows all three classes (codes 0=Normal, 1=Type_H, 2=Type_S)...
    le = LabelEncoder().fit(["Normal", "Type_H", "Type_S"])
    # ...but the model is only trained on codes 0 and 2 (Type_H absent), so
    # model.classes_ == [0, 2]. Labeling proba columns by le.classes_ (the old
    # bug) would mislabel column 1 as "Type_H" instead of "Type_S".
    X = pd.DataFrame(
        [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]], columns=FEATURES
    )
    model = KNeighborsClassifier(n_neighbors=1).fit(X, [0, 2])

    p = PredictionPipeline.__new__(PredictionPipeline)
    p.feature_columns = tuple(FEATURES)
    p.model = model
    p.label_encoder = le

    out = p.predict({c: 0 for c in FEATURES})
    # Only the two trained classes appear, correctly named by model order.
    assert set(out["probabilities"]) == {"Normal", "Type_S"}
    assert "Type_H" not in out["probabilities"]
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-6
    assert out["prediction"] == "Normal"
