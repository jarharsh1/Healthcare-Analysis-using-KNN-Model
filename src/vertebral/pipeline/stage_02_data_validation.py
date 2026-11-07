"""DVC stage: data validation. Aborts the pipeline if the schema check fails."""

from vertebral.components.data_validation import DataValidation
from vertebral.config.configuration import ConfigurationManager
from vertebral.logger import logger

STAGE = "Data Validation"


def run() -> None:
    config = ConfigurationManager().get_data_validation_config()
    passed = DataValidation(config).validate()
    if not passed:
        raise ValueError("Data validation failed; see status.json for details.")


if __name__ == "__main__":
    logger.info(">>> %s stage started <<<", STAGE)
    run()
    logger.info(">>> %s stage completed <<<\n", STAGE)
