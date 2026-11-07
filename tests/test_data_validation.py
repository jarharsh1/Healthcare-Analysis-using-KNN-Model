"""Unit tests for the schema-based data validation component."""

import pandas as pd

from vertebral.components.data_validation import DataValidation
from vertebral.entity.config_entity import DataValidationConfig
from vertebral.utils.common import load_json, read_yaml

SCHEMA = read_yaml("schema.yaml")


def _config(tmp_path, data_file):
    return DataValidationConfig(
        root_dir=tmp_path,
        data_file=data_file,
        status_file=tmp_path / "status.json",
        schema=SCHEMA,
    )


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "P_incidence": [63.03, 38.5],
            "P_tilt": [22.55, 16.96],
            "L_angle": [39.61, 35.11],
            "S_slope": [40.48, 21.54],
            "P_radius": [98.67, 127.63],
            "S_Degree": [-0.25, 7.99],
            "Class": ["Type_H", "Normal"],
        }
    )


def test_valid_data_passes(tmp_path):
    f = tmp_path / "ok.csv"
    _valid_frame().to_csv(f, index=False)
    assert DataValidation(_config(tmp_path, f)).validate() is True
    status = load_json(tmp_path / "status.json")
    assert status["validation_passed"] is True
    assert status["errors"] == []


def test_unknown_class_fails(tmp_path):
    df = _valid_frame()
    df.loc[0, "Class"] = "Type_X"
    f = tmp_path / "bad_class.csv"
    df.to_csv(f, index=False)
    assert DataValidation(_config(tmp_path, f)).validate() is False


def test_missing_column_fails(tmp_path):
    df = _valid_frame().drop(columns=["P_radius"])
    f = tmp_path / "missing.csv"
    df.to_csv(f, index=False)
    assert DataValidation(_config(tmp_path, f)).validate() is False


def test_missing_target_fails(tmp_path):
    df = _valid_frame()
    df.loc[0, "Class"] = None
    f = tmp_path / "missing_target.csv"
    df.to_csv(f, index=False)
    assert DataValidation(_config(tmp_path, f)).validate() is False
