"""Small filesystem/serialization helpers shared across components."""

import json
import os
from pathlib import Path
from typing import Any

import joblib
import yaml

from vertebral.logger import logger


def read_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    logger.info("Loaded YAML: %s", path)
    return content or {}


def create_directories(paths: list[str | Path]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)
        logger.info("Ensured directory: %s", p)


def save_json(path: str | Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info("Saved JSON: %s", path)


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_object(path: str | Path, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Saved object: %s", path)


def load_object(path: str | Path) -> Any:
    return joblib.load(path)
