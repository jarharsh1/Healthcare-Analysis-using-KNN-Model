"""DVC stage: data transformation."""

from vertebral.components.data_transformation import DataTransformation
from vertebral.config.configuration import ConfigurationManager
from vertebral.logger import logger

STAGE = "Data Transformation"


def run() -> None:
    config = ConfigurationManager().get_data_transformation_config()
    DataTransformation(config).transform()


if __name__ == "__main__":
    logger.info(">>> %s stage started <<<", STAGE)
    run()
    logger.info(">>> %s stage completed <<<\n", STAGE)
