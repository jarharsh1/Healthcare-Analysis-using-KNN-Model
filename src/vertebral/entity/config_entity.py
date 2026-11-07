"""Typed configuration objects for each pipeline stage."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_files: list
    local_data_file: Path
    target_column: str
    label_aliases: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    data_file: Path
    status_file: Path
    schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_file: Path
    preprocessor_file: Path
    label_encoder_file: Path
    train_file: Path
    test_file: Path
    target_column: str
    numeric_columns: list = field(default_factory=list)
    test_size: float = 0.2
    random_state: int = 42


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_file: Path
    preprocessor_file: Path
    model_file: Path
    target_column: str
    n_neighbors: int
    weights: str
    metric: str
    p: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_file: Path
    model_file: Path
    label_encoder_file: Path
    metrics_file: Path
    target_column: str
    average: str
    mlflow_uri: str
    all_params: dict
