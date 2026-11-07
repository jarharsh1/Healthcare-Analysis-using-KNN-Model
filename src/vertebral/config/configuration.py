"""Read the three YAML files and hand out typed config objects per stage."""

from pathlib import Path

from vertebral.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
)
from vertebral.utils.common import create_directories, read_yaml

CONFIG_FILE = "config/config.yaml"
PARAMS_FILE = "params.yaml"
SCHEMA_FILE = "schema.yaml"


class ConfigurationManager:
    def __init__(
        self,
        config_file: str = CONFIG_FILE,
        params_file: str = PARAMS_FILE,
        schema_file: str = SCHEMA_FILE,
    ):
        self.config = read_yaml(config_file)
        self.params = read_yaml(params_file)
        self.schema = read_yaml(schema_file)
        create_directories([self.config["artifacts_root"]])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        c = self.config["data_ingestion"]
        create_directories([c["root_dir"]])
        return DataIngestionConfig(
            root_dir=Path(c["root_dir"]),
            source_files=[Path(f) for f in c["source_files"]],
            local_data_file=Path(c["local_data_file"]),
            target_column=self.schema["target_column"],
            label_aliases=self.schema.get("label_aliases", {}),
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        c = self.config["data_validation"]
        create_directories([c["root_dir"]])
        return DataValidationConfig(
            root_dir=Path(c["root_dir"]),
            data_file=Path(c["data_file"]),
            status_file=Path(c["status_file"]),
            schema=self.schema,
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        c = self.config["data_transformation"]
        p = self.params["data_transformation"]
        create_directories([c["root_dir"]])
        return DataTransformationConfig(
            root_dir=Path(c["root_dir"]),
            data_file=Path(c["data_file"]),
            preprocessor_file=Path(c["preprocessor_file"]),
            label_encoder_file=Path(c["label_encoder_file"]),
            train_file=Path(c["train_file"]),
            test_file=Path(c["test_file"]),
            target_column=self.schema["target_column"],
            numeric_columns=self.schema["numeric_columns"],
            test_size=p["test_size"],
            random_state=p["random_state"],
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        c = self.config["model_trainer"]
        p = self.params["model_trainer"]
        create_directories([c["root_dir"]])
        return ModelTrainerConfig(
            root_dir=Path(c["root_dir"]),
            train_file=Path(c["train_file"]),
            preprocessor_file=Path(c["preprocessor_file"]),
            model_file=Path(c["model_file"]),
            target_column=self.schema["target_column"],
            n_neighbors=p["n_neighbors"],
            weights=p["weights"],
            metric=p["metric"],
            p=p["p"],
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        c = self.config["model_evaluation"]
        p = self.params["model_evaluation"]
        create_directories([c["root_dir"]])
        return ModelEvaluationConfig(
            root_dir=Path(c["root_dir"]),
            test_file=Path(c["test_file"]),
            model_file=Path(c["model_file"]),
            label_encoder_file=Path(c["label_encoder_file"]),
            metrics_file=Path(c["metrics_file"]),
            target_column=self.schema["target_column"],
            average=p["average"],
            mlflow_uri=c.get("mlflow_uri", ""),
            all_params=self.params,
        )
