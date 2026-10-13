
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "artifacts" / "crop_model.joblib"

app = FastAPI(title="AgriSense-MLOps")

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        "Trained model not found. Run python src/train_model.py first."
    )

bundle = joblib.load(MODEL_FILE)
model = bundle["model"]
features = bundle["features"]


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AgriSense-MLOps</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 650px;
                margin: 40px auto;
                padding: 20px;
                background: #f4f8f2;
                color: #243424;
            }
            .panel {
                background: white;
                padding: 24px;
                border-radius: 12px;
                box-shadow: 0 2px 10px #0001;
            }
            label { display: block; margin-top: 14px; }
            input {
                box-sizing: border-box;
                width: 100%;
                padding: 10px;
                margin-top: 5px;
            }
            button {
                margin-top: 22px;
                padding: 12px 18px;
                background: #287a3d;
                color: white;
                border: 0;
                border-radius: 6px;
                cursor: pointer;
            }
            .note { color: #555; font-size: 14px; }
        </style>
    </head>
    <body>
        <div class="panel">
            <h1>AgriSense-MLOps</h1>
            <p>Crop Health Prediction Demo</p>
            <p class="note">
                Enter sample sensor readings to test the trained model.
                This demonstration uses synthetic data, not real farm guidance.
            </p>

            <form action="/predict" method="post">
                <label>Temperature</label>
                <input type="number" name="temperature"
                       step="any" required value="25">

                <label>Humidity</label>
                <input type="number" name="humidity"
                       step="any" required value="65">

                <label>Soil Moisture</label>
                <input type="number" name="soil_moisture"
                       step="any" required value="55">

                <label>Rainfall</label>
                <input type="number" name="rainfall"
                       step="any" required value="100">

                <button type="submit">Predict Crop Condition</button>
            </form>
        </div>
    </body>
    </html>
    """


@app.post("/predict", response_class=HTMLResponse)
def predict(
    temperature: float = Form(...),
    humidity: float = Form(...),
    soil_moisture: float = Form(...),
    rainfall: float = Form(...),
):
    values = {
        "temperature": temperature,
        "humidity": humidity,
        "soil_moisture": soil_moisture,
        "rainfall": rainfall,
    }

    if not all(pd.notna(value) and abs(value) != float("inf")
               for value in values.values()):
        return HTMLResponse(
            "<h2>Invalid input values.</h2><a href='/'>Go back</a>",
            status_code=400,
        )

    sample = pd.DataFrame([values], columns=features)
    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    scores = {
        label: float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }
    healthy_score = scores.get("Healthy", 0.0) * 100
    stressed_score = scores.get("Stressed", 0.0) * 100

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prediction Result</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 650px;
                margin: 40px auto;
                padding: 20px;
                background: #f4f8f2;
                color: #243424;
            }}
            .panel {{
                background: white;
                padding: 24px;
                border-radius: 12px;
            }}
            a {{ color: #287a3d; }}
        </style>
    </head>
    <body>
        <div class="panel">
            <h1>Prediction Result</h1>
            <h2>Predicted condition: {prediction}</h2>
            <p>Healthy score: {healthy_score:.2f}%</p>
            <p>Stressed score: {stressed_score:.2f}%</p>
            <p>This is a model prediction based on synthetic training data.</p>
            <a href="/">Try another prediction</a>
        </div>
    </body>
    </html>
    """
