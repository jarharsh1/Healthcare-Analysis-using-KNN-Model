"""DVC stage: data transformation.

Before transforming, this stage re-checks the validation gate: it reads
``status.json`` and refuses to run unless ``validation_passed`` is true. The full
pipeline already stops at stage 2 on failure, but this guard means invoking stage
3 directly (or out of order) still respects the gate rather than transforming
data that never passed validation.
"""

from vertebral.components.data_transformation import DataTransformation
from vertebral.config.configuration import ConfigurationManager
from vertebral.logger import logger
from vertebral.utils.common import load_json

STAGE = "Data Transformation"


def _require_validation_passed(manager: ConfigurationManager) -> None:
    status_file = manager.get_data_validation_config().status_file
    if not status_file.exists():
        raise FileNotFoundError(
            f"Validation status not found at {status_file}; run data validation first."
        )
    status = load_json(status_file)
    if not status.get("validation_passed"):
        raise ValueError(
            "Data validation did not pass; refusing to transform. "
            "See status.json for details."
        )


def run() -> None:
    manager = ConfigurationManager()
    _require_validation_passed(manager)
    config = manager.get_data_transformation_config()
    DataTransformation(config).transform()


if __name__ == "__main__":
    logger.info(">>> %s stage started <<<", STAGE)
    run()
    logger.info(">>> %s stage completed <<<\n", STAGE)
