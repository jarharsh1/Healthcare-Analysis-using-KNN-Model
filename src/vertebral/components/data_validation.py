"""Stage 2 - Data validation.

Checks the ingested frame against schema.yaml: column set matches, the six
biomechanics features are numeric, no fully-null columns, and the target is
present, complete, and restricted to the three known classes. Writes status.json.
"""

import pandas as pd

from vertebral.entity.config_entity import DataValidationConfig
from vertebral.logger import logger
from vertebral.utils.common import save_json


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate(self) -> bool:
        df = pd.read_csv(self.config.data_file)
        schema = self.config.schema
        target = schema["target_column"]
        expected = set(schema["columns"])
        numeric_cols = schema["numeric_columns"]
        valid_classes = set(schema["target_classes"])

        errors: list[str] = []

        # 1. Column set matches schema exactly.
        actual = set(df.columns)
        missing = expected - actual
        extra = actual - expected
        if missing:
            errors.append(f"Missing columns: {sorted(missing)}")
        if extra:
            errors.append(f"Unexpected columns: {sorted(extra)}")

        # 2. Feature columns must be numeric.
        for col in numeric_cols:
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' is not numeric ({df[col].dtype})")

        # 3. No completely empty columns.
        for col in df.columns:
            if df[col].isna().all():
                errors.append(f"Column '{col}' is entirely null")

        # 4. Target present, complete, and only known classes.
        if target in df.columns:
            n_missing = int(df[target].isna().sum())
            if n_missing:
                errors.append(f"Target '{target}' has {n_missing} missing value(s)")
            labels = set(df[target].dropna().unique())
            unknown = labels - valid_classes
            if unknown:
                errors.append(f"Target '{target}' has unknown classes: {sorted(unknown)}")

        status = {
            "validation_passed": len(errors) == 0,
            "n_rows": len(df),
            "n_columns": int(df.shape[1]),
            "class_counts": (
                df[target].value_counts().to_dict() if target in df.columns else {}
            ),
            "errors": errors,
        }
        save_json(self.config.status_file, status)

        if errors:
            logger.error("Data validation FAILED: %s", errors)
        else:
            logger.info("Data validation passed (%d rows).", len(df))
        return status["validation_passed"]
