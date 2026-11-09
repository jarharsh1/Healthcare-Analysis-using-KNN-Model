"""Inference pipeline used by the serving layer.

Loads the trained serving pipeline and the label encoder, predicts the class
and decodes the integer label back to its name (Normal / Type_H / Type_S),
and returns per-class probabilities.
"""

from functools import lru_cache

import pandas as pd

from vertebral.config.configuration import ConfigurationManager
from vertebral.utils.common import load_object


@lru_cache(maxsize=1)
def _load_model():
    model_file = ConfigurationManager().get_model_trainer_config().model_file
    return load_object(model_file)


@lru_cache(maxsize=1)
def _load_label_encoder():
    le_file = ConfigurationManager().get_data_transformation_config().label_encoder_file
    return load_object(le_file)


@lru_cache(maxsize=1)
def _feature_columns() -> tuple:
    return tuple(ConfigurationManager().get_data_transformation_config().numeric_columns)


class PredictionPipeline:
    def __init__(self):
        self.model = _load_model()
        self.label_encoder = _load_label_encoder()
        self.feature_columns = _feature_columns()

    def predict(self, features: dict) -> dict:
        missing = [c for c in self.feature_columns if c not in features]
        if missing:
            raise ValueError(f"Missing required feature field(s): {missing}")
        # Reindex to the training feature order (extra keys ignored).
        df = pd.DataFrame([features]).reindex(columns=list(self.feature_columns))
        pred_idx = int(self.model.predict(df)[0])
        pred_class = str(self.label_encoder.inverse_transform([pred_idx])[0])

        proba = self.model.predict_proba(df)[0]
        # Label each probability by decoding the MODEL's own class order
        # (model.classes_), not label_encoder.classes_ — these only coincide
        # when every class is present in training, so decode the model's order
        # to stay correct even if a class is absent.
        class_names = [
            str(c) for c in self.label_encoder.inverse_transform(self.model.classes_)
        ]
        probabilities = {
            name: round(float(p), 4) for name, p in zip(class_names, proba)
        }
        return {
            "prediction": pred_class,
            "probabilities": probabilities,
        }
