"""Stage 3 - Data transformation.

Label-encodes the 3-class target (saving the encoder so predictions can be
decoded back to class names), builds a numeric ColumnTransformer (impute +
scale — scaling matters for the distance-based KNN), and writes a stratified
train/test split plus the preprocessor fitted on the training split only.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from vertebral.entity.config_entity import DataTransformationConfig
from vertebral.logger import logger
from vertebral.utils.common import save_object


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def _build_preprocessor(self) -> ColumnTransformer:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        return ColumnTransformer(
            transformers=[("num", numeric_pipe, self.config.numeric_columns)]
        )

    def transform(self) -> tuple[str, str, str]:
        df = pd.read_csv(self.config.data_file)
        target = self.config.target_column

        # Drop rows with a missing target rather than inventing a class.
        n_missing = int(df[target].isna().sum())
        if n_missing:
            logger.warning("Dropping %d row(s) with a missing target.", n_missing)
            df = df.dropna(subset=[target])

        # Encode the string classes to integers; persist the encoder for decoding.
        label_encoder = LabelEncoder()
        df[target] = label_encoder.fit_transform(df[target].astype(str))
        save_object(self.config.label_encoder_file, label_encoder)

        train_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=df[target],
        )

        preprocessor = self._build_preprocessor()
        preprocessor.fit(train_df[self.config.numeric_columns])

        save_object(self.config.preprocessor_file, preprocessor)
        train_df.to_csv(self.config.train_file, index=False)
        test_df.to_csv(self.config.test_file, index=False)

        logger.info(
            "Transformation done: train=%d test=%d; classes=%s",
            len(train_df),
            len(test_df),
            list(label_encoder.classes_),
        )
        return (
            str(self.config.train_file),
            str(self.config.test_file),
            str(self.config.preprocessor_file),
        )
