
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Find the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# File locations
DATA_FILE = BASE_DIR / "data" / "crop_data.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_FILE = ARTIFACTS_DIR / "crop_model.joblib"

# Input columns used for prediction
FEATURES = [
    "temperature",
    "humidity",
    "soil_moisture",
    "rainfall",
]

TARGET = "crop_condition"

def train_model():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Dataset not found. Run python src/generate_data.py first."
        )

    data = pd.read_csv(DATA_FILE)

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    # Set up MLflow tracking  
    mlflow.set_tracking_uri(
    f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}"
    )
    mlflow.set_experiment("AgriSense-Crop-Health")


    with mlflow.start_run():
        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Record experiment settings and results
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("test_accuracy", float(accuracy))
        mlflow.log_metric("training_rows", len(X_train))
        mlflow.log_metric("testing_rows", len(X_test))

        # Save the trained model
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        model_bundle = {
            "model": model,
            "features": FEATURES,
            "labels": sorted(y.unique().tolist()),
        }
        joblib.dump(model_bundle, MODEL_FILE)

        # Display the results
        print("Model training completed successfully!")
        print(f"Training rows: {len(X_train)}")
        print(f"Testing rows: {len(X_test)}")
        print(f"Test accuracy: {accuracy:.2%}")
        print("\nClassification report:")
        print(classification_report(y_test, predictions))
        print(f"Saved model to: {MODEL_FILE}")
        print("MLflow experiment: AgriSense-Crop-Health")
        print(f"MLflow records saved in: {BASE_DIR / 'mlruns'}")


if __name__ == "__main__":
    train_model()



