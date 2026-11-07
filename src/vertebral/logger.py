"""Central logging: stdout (for CI/Docker) plus a rotating file under logs/."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "running_logs.log")
os.makedirs(LOG_DIR, exist_ok=True)

_FORMAT = "[%(asctime)s] %(levelname)s %(name)s (%(module)s:%(lineno)d) - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=_FORMAT,
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3),
        logging.StreamHandler(sys.stdout),
    ],
)


def get_logger(name: str = "vertebral") -> logging.Logger:
    return logging.getLogger(name)


logger = get_logger()
