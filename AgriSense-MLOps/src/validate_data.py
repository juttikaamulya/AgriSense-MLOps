
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "crop_data.csv"

REQUIRED_COLUMNS = [
    "temperature",
    "humidity",
    "soil_moisture",
    "rainfall",
    "crop_condition",
]

NUMERIC_COLUMNS = [
    "temperature",
    "humidity",
    "soil_moisture",
    "rainfall",
]

ALLOWED_LABELS = {"Healthy", "Stressed"}


def validate_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}. "
            "Run python src/generate_data.py first."
        )

    data = pd.read_csv(DATA_FILE)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if data.empty:
        raise ValueError("The dataset is empty.")

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    if data[NUMERIC_COLUMNS].isna().any().any():
        raise ValueError(
            "Some numeric values are missing or invalid."
        )

    if not data[NUMERIC_COLUMNS].map(
        lambda value: pd.notna(value) and abs(value) != float("inf")
    ).all().all():
        raise ValueError("Infinite numeric values were found.")

    labels = set(data["crop_condition"].dropna().astype(str).str.strip())
    unexpected_labels = labels - ALLOWED_LABELS

    if data["crop_condition"].isna().any():
        raise ValueError("Some crop condition labels are missing.")

    if unexpected_labels:
        raise ValueError(
            f"Unexpected crop labels: {unexpected_labels}"
        )

    if len(labels) < 2:
        raise ValueError(
            "The dataset needs at least two crop condition labels."
        )

    print("Dataset validation passed!")
    print(f"Number of rows: {len(data)}")
    print(f"Number of columns: {len(data.columns)}")
    print("\nMissing values per column:")
    print(data[REQUIRED_COLUMNS].isna().sum())
    print("\nCrop condition counts:")
    print(data["crop_condition"].value_counts())


if __name__ == "__main__":
    validate_dataset()
