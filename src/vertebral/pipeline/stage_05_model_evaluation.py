"""DVC stage: model evaluation."""

from vertebral.components.model_evaluation import ModelEvaluation
from vertebral.config.configuration import ConfigurationManager
from vertebral.logger import logger

STAGE = "Model Evaluation"


def run() -> None:
    config = ConfigurationManager().get_model_evaluation_config()
    ModelEvaluation(config).evaluate()


if __name__ == "__main__":
    logger.info(">>> %s stage started <<<", STAGE)
    run()
    logger.info(">>> %s stage completed <<<\n", STAGE)
