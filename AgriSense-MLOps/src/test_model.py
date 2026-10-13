
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "artifacts" / "crop_model.joblib"


def test_saved_model():
    # Check that the trained model exists
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Trained model not found. Run python src/train_model.py first."
        )

    # Load the saved model
    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    features = bundle["features"]
    labels = bundle["labels"]

    # Example sensor readings (synthetic test values)
    sample = pd.DataFrame(
        [
            {
                "temperature": 25.0,
                "humidity": 65.0,
                "soil_moisture": 55.0,
                "rainfall": 100.0,
            }
        ],
        columns=features,
    )

    # Make a prediction
    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]

    print("Saved model loaded successfully!")
    print(f"Input readings:\n{sample.to_string(index=False)}")
    print(f"\nPredicted crop condition: {prediction}")
    print("\nPrediction scores:")

    for label, probability in zip(model.classes_, probabilities):
        print(f"{label}: {probability:.2%}")

    print(f"\nAllowed labels: {labels}")


if __name__ == "__main__":
    test_saved_model()
