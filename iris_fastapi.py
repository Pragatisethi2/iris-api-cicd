# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="🌸 Iris Classifier API")

# Load model
model = joblib.load("model.joblib")

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API!"}

@app.post("/predict/")
def predict_species(data: IrisInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return {
        "predicted_class": prediction
    }

@app.get("/version")
def get_version():
    return {"version": "2.0", "message": "Updated via CD pipeline!"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Iris ML API",
        "version": "1.0",
        "deployment": "Kubernetes + Docker",
        "timestamp": "2025-07-26"
    }

@app.get("/info")
def api_info():
    return {
        "api_name": "Iris Species Classifier",
        "model_type": "Decision Tree",
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "classes": ["setosa", "versicolor", "virginica"],
        "author": "Pragati Sethi",
        "deployment": "GKE with CI/CD"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Iris ML API",
        "version": "1.0",
        "deployment": "Kubernetes + Docker",
        "timestamp": "2025-07-26"
    }

@app.get("/info")
def api_info():
    return {
        "api_name": "Iris Species Classifier",
        "model_type": "Decision Tree",
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "classes": ["setosa", "versicolor", "virginica"],
        "author": "Pragati Sethi",
        "deployment": "GKE with CI/CD"
    }
