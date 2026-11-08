"""Stage 4 - Model training.

Loads the fitted preprocessor and fits a KNeighborsClassifier (honouring the
original KNN model) on the training split, bundled with the preprocessor into a
single serving Pipeline so serving loads one object and accepts raw feature rows.
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from vertebral.entity.config_entity import ModelTrainerConfig
from vertebral.logger import logger
from vertebral.utils.common import load_object, save_object


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self) -> str:
        train_df = pd.read_csv(self.config.train_file)
        target = self.config.target_column
        X = train_df.drop(columns=[target])
        y = train_df[target]

        preprocessor = load_object(self.config.preprocessor_file)

        model = KNeighborsClassifier(
            n_neighbors=self.config.n_neighbors,
            weights=self.config.weights,
            metric=self.config.metric,
            p=self.config.p,
        )

        # NOTE: this .fit() refits the preprocessor on the training rows (the
        # same train-only data it was already fitted on in transformation), so
        # the served preprocessor has no test leakage. The standalone
        # preprocessor.joblib from the transformation stage is therefore not used
        # at serving time; it is kept only for lineage / inspection.
        serving_pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", model)]
        )
        serving_pipeline.fit(X, y)

        save_object(self.config.model_file, serving_pipeline)
        logger.info(
            "Trained KNN(n_neighbors=%d, weights=%s) -> %s",
            self.config.n_neighbors,
            self.config.weights,
            self.config.model_file,
        )
        return str(self.config.model_file)
