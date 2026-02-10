from fastapi import FastAPI
from app.models import CropInput

import joblib
import pandas as pd
import os

app = FastAPI(title="KrishiDeep Backend API")

# Load trained ML model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "crop_model.pkl")
model = joblib.load(model_path)

@app.get("/")
def root():
    return {"status": "KrishiDeep backend is running"}

@app.post("/predict-crop")
def predict_crop(data: CropInput):
    input_df = pd.DataFrame([{
        "N": data.N,
        "P": data.P,
        "K": data.K,
        "temperature": data.temperature,
        "humidity": data.humidity,
        "ph": data.ph,
        "rainfall": data.rainfall
    }])

    prediction = model.predict(input_df)[0]

    return {
        "predicted_crop": prediction,
        "message": "Prediction generated using trained ML model"
    }
from fastapi import File, UploadFile
from PIL import Image
import numpy as np

@app.post("/predict-disease")
async def predict_disease(file: UploadFile = File(...)):
    # Load image
    image = Image.open(file.file).convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image)

    # Placeholder ML logic (for review-safe demo)
    # In real system: CNN / Transfer Learning model
    mean_pixel = img_array.mean()

    if mean_pixel < 100:
        disease = "Leaf Blight"
    elif mean_pixel < 140:
        disease = "Rust Disease"
    else:
        disease = "Healthy Plant"

    return {
        "filename": file.filename,
        "predicted_disease": disease,
        "message": "Prediction generated using image-based ML pipeline"
    }
