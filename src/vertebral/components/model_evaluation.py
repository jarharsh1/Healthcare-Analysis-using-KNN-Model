"""Stage 5 - Model evaluation.

Scores the serving pipeline on the held-out test split with macro-averaged
multiclass metrics and one-vs-rest ROC-AUC, writes metrics.json, and logs
params + metrics + model to MLflow (optional import; local mlruns/).
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from vertebral.entity.config_entity import ModelEvaluationConfig
from vertebral.logger import logger
from vertebral.utils.common import load_object, save_json

try:
    import mlflow
    import mlflow.sklearn

    _MLFLOW_AVAILABLE = True
except ImportError:  # MLflow is optional; evaluation still writes metrics.json
    _MLFLOW_AVAILABLE = False


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self) -> dict:
        test_df = pd.read_csv(self.config.test_file)
        target = self.config.target_column
        X = test_df.drop(columns=[target])
        y = test_df[target]
        avg = self.config.average

        model = load_object(self.config.model_file)
        preds = model.predict(X)

        metrics = {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, average=avg, zero_division=0)),
            "recall": float(recall_score(y, preds, average=avg, zero_division=0)),
            "f1": float(f1_score(y, preds, average=avg, zero_division=0)),
        }

        # Multiclass ROC-AUC (one-vs-rest) when probabilities are available.
        # Always record the key: null means "not computable" (e.g. a class is
        # absent from the test split), distinct from the key being missing.
        try:
            proba = model.predict_proba(X)
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(y, proba, multi_class="ovr", average=avg)
            )
        except Exception as exc:  # pragma: no cover - metric best-effort
            logger.warning("Could not compute ROC-AUC: %s", exc)
            metrics["roc_auc_ovr"] = None

        save_json(self.config.metrics_file, metrics)
        logger.info("Evaluation metrics: %s", metrics)

        self._log_to_mlflow(metrics, model)
        return metrics

    def _log_to_mlflow(self, metrics: dict, model) -> None:
        if not _MLFLOW_AVAILABLE:
            logger.warning("MLflow not installed; skipping experiment logging.")
            return
        if self.config.mlflow_uri:
            mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment("vertebral-column")

        with mlflow.start_run():
            for section, params in self.config.all_params.items():
                if isinstance(params, dict):
                    for k, v in params.items():
                        mlflow.log_param(f"{section}.{k}", v)
            # MLflow metrics must be numeric; drop any None (e.g. undefined AUC).
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
            try:
                mlflow.sklearn.log_model(model, name="model")
            except Exception as exc:  # pragma: no cover - logging best-effort
                logger.warning("Could not log model to MLflow: %s", exc)
        logger.info("Logged run to MLflow experiment 'vertebral-column'.")
