import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Global Forest Fire Prediction System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "fire_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

class PredictionInput(BaseModel):
    temp: float
    wind: float
    humidity: float
    vegetation: float = 0.5
    soil_moisture: float = 0.3

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Wildfire Prediction API is running",
        "version": "2.0.0"
    }

@app.post("/predict")
def predict(data: PredictionInput):
    features = np.array([[data.temp, data.wind, data.humidity, data.vegetation, data.soil_moisture]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    risk_score = float(probability[1] * 100)

    if risk_score < 30:
        status = "Low Risk"
    elif risk_score < 70:
        status = "Moderate Risk"
    else:
        status = "High Risk"

    return {
        "fire_detected": bool(prediction),
        "risk_percentage": f"{risk_score:.2f}%",
        "status": status,
        "recommendation": "Monitor local weather alerts" if prediction else "Conditions are stable"
    }