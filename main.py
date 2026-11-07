"""Run the full training pipeline end-to-end (outside DVC).

    python main.py

Each stage is also runnable on its own via ``python -m
vertebral.pipeline.stage_0X_...`` and wired into dvc.yaml for reproducibility.
"""

from vertebral.logger import logger
from vertebral.pipeline import (
    stage_01_data_ingestion,
    stage_02_data_validation,
    stage_03_data_transformation,
    stage_04_model_trainer,
    stage_05_model_evaluation,
)

STAGES = [
    (stage_01_data_ingestion.STAGE, stage_01_data_ingestion.run),
    (stage_02_data_validation.STAGE, stage_02_data_validation.run),
    (stage_03_data_transformation.STAGE, stage_03_data_transformation.run),
    (stage_04_model_trainer.STAGE, stage_04_model_trainer.run),
    (stage_05_model_evaluation.STAGE, stage_05_model_evaluation.run),
]


def main() -> None:
    for name, run in STAGES:
        logger.info(">>> %s stage started <<<", name)
        run()
        logger.info(">>> %s stage completed <<<\n", name)
    logger.info("Full pipeline completed successfully.")


if __name__ == "__main__":
    main()
