"""DVC stage: data ingestion."""

from vertebral.components.data_ingestion import DataIngestion
from vertebral.config.configuration import ConfigurationManager
from vertebral.logger import logger

STAGE = "Data Ingestion"


def run() -> None:
    config = ConfigurationManager().get_data_ingestion_config()
    DataIngestion(config).ingest()


if __name__ == "__main__":
    logger.info(">>> %s stage started <<<", STAGE)
    run()
    logger.info(">>> %s stage completed <<<\n", STAGE)
