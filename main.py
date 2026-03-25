# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Diabetes Prediction API")

# Load model once at startup
try:
    model = joblib.load("diabetes_model.pkl")
    print("Model loaded successfully")
except Exception as e:
    print("Model loading failed:", e)
    model = None


class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    BMI: float
    Age: int


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict")
def predict(data: DiabetesInput):
    if model is None:
        return {"error": "Model not loaded"}

    try:
        input_data = np.array([[ 
            data.Pregnancies,
            data.Glucose,
            data.BloodPressure,
            data.BMI,
            data.Age
        ]])

        prediction = model.predict(input_data)[0]

        return {
            "diabetic": bool(prediction)
        }

    except Exception as e:
        return {"error": str(e)}
    
# # main.py
# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import numpy as np

# app = FastAPI()
# model = joblib.load("diabetes_model.pkl")

# class DiabetesInput(BaseModel):
#     Pregnancies: int
#     Glucose: float
#     BloodPressure: float
#     BMI: float
#     Age: int

# @app.get("/")
# def read_root():
#     return {"message": "Diabetes Prediction API is live"}

# @app.post("/predict")
# def predict(data: DiabetesInput):
#     input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, data.BMI, data.Age]])
#     prediction = model.predict(input_data)[0]
#     return {"diabetic": bool(prediction)}
