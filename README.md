# 🧠 Wildfire Risk AI Backend

The AI engine and API layer for the **Wildfire Risk Intelligence** system. This repository hosts the Machine Learning model logic and provides the prediction service for the frontend dashboard.

### 🌍 Live System
The complete end-to-end application can be accessed here:
👉 **[https://ww3carla.github.io/Forest-Fire-Prediction-Frontend/](https://ww3carla.github.io/Forest-Fire-Prediction-Frontend/)**


## 🛰️ Overview
Wildfire Risk AI Backend is a Python-based service that transforms meteorological data into actionable wildfire risk intelligence. 

It handles:
* **Inference**: Running the pre-trained Machine Learning model.
* **API Service**: Providing a fast and reliable endpoint for the React frontend.
* **Logic**: Calculating risk classifications and safety recommendations.


## 🚀 Backend Features
* **FastAPI Framework**: High-performance, asynchronous Python API.
* **AI Prediction Engine**: Powered by a custom-trained Scikit-learn model.
* **Deployment**: Hosted on Hugging Face Spaces for seamless AI integration.
* **Real-time Processing**: Fast response times for instant dashboard updates.


## 🛠 Tech Stack
* **Language:** Python
* **Framework:** FastAPI
* **ML Libraries:** Scikit-learn, Pandas, NumPy
* **Server:** Uvicorn
* **Deployment:** Hugging Face Spaces / Docker

## 📊 How It Works (AI Logic)
Request: The backend receives Temperature, Humidity, and Wind Speed.

Feature Engineering: Data is normalized and prepared for the model.

Prediction: The Random Forest model estimates the fire probability.

Classification: Based on the percentage, the backend assigns a status (Low, Moderate, High, Extreme).

Response: Results are sent back to the React dashboard for visualization.

## 📁 Project Structure
app.py - Main FastAPI application and routing.

model.pkl - The trained Machine Learning model.

requirements.txt - Python dependencies (FastAPI, Scikit-learn, etc.).

👩‍💻 Author
Carla Bozintan
