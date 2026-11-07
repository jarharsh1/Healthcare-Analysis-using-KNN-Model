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


class PredictionPipeline:
    def __init__(self):
        self.model = _load_model()
        self.label_encoder = _load_label_encoder()

    def predict(self, features: dict) -> dict:
        df = pd.DataFrame([features])
        pred_idx = int(self.model.predict(df)[0])
        pred_class = str(self.label_encoder.inverse_transform([pred_idx])[0])

        proba = self.model.predict_proba(df)[0]
        class_names = [str(c) for c in self.label_encoder.classes_]
        probabilities = {
            name: round(float(p), 4) for name, p in zip(class_names, proba)
        }
        return {
            "prediction": pred_class,
            "probabilities": probabilities,
        }
