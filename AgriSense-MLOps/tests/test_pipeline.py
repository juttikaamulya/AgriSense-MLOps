
from pathlib import Path

import joblib
import pandas as pd
import pytest


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "crop_data.csv"
MODEL_FILE = BASE_DIR / "artifacts" / "crop_model.joblib"

FEATURES = [
    "temperature",
    "humidity",
    "soil_moisture",
    "rainfall",
]

ALLOWED_LABELS = {"Healthy", "Stressed"}


def test_dataset_exists():
    assert DATA_FILE.exists(), "Dataset file is missing."


def test_dataset_has_required_columns():
    data = pd.read_csv(DATA_FILE)

    required_columns = FEATURES + ["crop_condition"]

    for column in required_columns:
        assert column in data.columns, f"Missing column: {column}"


def test_dataset_has_no_missing_values():
    data = pd.read_csv(DATA_FILE)

    assert not data[FEATURES + ["crop_condition"]].isna().any().any()


def test_labels_are_valid():
    data = pd.read_csv(DATA_FILE)

    assert set(data["crop_condition"].unique()).issubset(ALLOWED_LABELS)
    assert data["crop_condition"].nunique() == 2


def test_saved_model_exists():
    assert MODEL_FILE.exists(), "Trained model file is missing."


def test_model_predicts_valid_label():
    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]

    sample = pd.DataFrame(
        [{
            "temperature": 25.0,
            "humidity": 65.0,
            "soil_moisture": 55.0,
            "rainfall": 100.0,
        }],
        columns=FEATURES,
    )

    prediction = model.predict(sample)[0]

    assert prediction in ALLOWED_LABELS


def test_model_probabilities_are_valid():
    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]

    sample = pd.DataFrame(
        [{
            "temperature": 25.0,
            "humidity": 65.0,
            "soil_moisture": 55.0,
            "rainfall": 100.0,
        }],
        columns=FEATURES,
    )

    probabilities = model.predict_proba(sample)[0]

    assert all(0 <= probability <= 1 for probability in probabilities)
    assert sum(probabilities) == pytest.approx(1.0)
