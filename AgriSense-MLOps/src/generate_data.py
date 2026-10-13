
from pathlib import Path

import numpy as np
import pandas as pd


# Find the main project folder automatically
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "crop_data.csv"


def generate_dataset(rows=1000):
    """Create a synthetic crop health dataset."""

    rng = np.random.default_rng(42)

    temperature = rng.uniform(15, 45, rows)
    humidity = rng.uniform(20, 100, rows)
    soil_moisture = rng.uniform(5, 95, rows)
    rainfall = rng.uniform(0, 150, rows)

    # Simple example rules for creating learning labels
    stressed = (
        ((temperature > 35) & (soil_moisture < 35))
        | ((soil_moisture < 15) & (rainfall < 20))
        | (temperature > 41)
    )

    crop_condition = np.where(
        stressed,
        "Stressed",
        "Healthy"
    )

    dataset = pd.DataFrame({
        "temperature": temperature.round(2),
        "humidity": humidity.round(2),
        "soil_moisture": soil_moisture.round(2),
        "rainfall": rainfall.round(2),
        "crop_condition": crop_condition,
    })

    return dataset


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = generate_dataset()

    dataset.to_csv(DATA_FILE, index=False)

    print("Dataset created successfully!")
    print(f"Total rows: {len(dataset)}")
    print(f"Saved location: {DATA_FILE}")
    print("\nFirst 5 rows:")
    print(dataset.head())
    print("\nCrop condition counts:")
    print(dataset["crop_condition"].value_counts())


if __name__ == "__main__":
    main()
