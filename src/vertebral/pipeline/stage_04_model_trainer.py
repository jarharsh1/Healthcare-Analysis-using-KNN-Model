"""DVC stage: model training."""

from vertebral.components.model_trainer import ModelTrainer
from vertebral.config.configuration import ConfigurationManager
from vertebral.logger import logger

STAGE = "Model Trainer"


def run() -> None:
    config = ConfigurationManager().get_model_trainer_config()
    ModelTrainer(config).train()


if __name__ == "__main__":
    logger.info(">>> %s stage started <<<", STAGE)
    run()
    logger.info(">>> %s stage completed <<<\n", STAGE)
