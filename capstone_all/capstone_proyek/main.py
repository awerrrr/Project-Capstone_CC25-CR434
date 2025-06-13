from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import tensorflow_decision_forests as tfdf
import ydf  # successor of TFDF

# Inisialisasi FastAPI
app = FastAPI(title="Laptop Price Prediction API")

# Load model TFDF
model = ydf.from_tensorflow_decision_forests("tfdf_model_laptop")

# Tambahkan CORS agar bisa diakses dari HTML/JS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti "*" dengan asal pasti jika perlu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schema input untuk prediksi
class LaptopSpecs(BaseModel):
    Ram: int
    Weight: float
    SSD: int
    TypeName_enc: int
    OpSys_enc: int

# Endpoint prediksi
@app.post("/predict")
def predict_price(specs: LaptopSpecs):
    input_df = pd.DataFrame([specs.dict()])
    print("📥 Input:", input_df)

    prediction = model.predict(input_df)[0]  # satu baris output
    print("📤 Prediction:", prediction)

    return {"predicted_price_idr": float(round(prediction, 2))}

# Health check
@app.get("/health")
def health_check():
    return {"status": "ok"}
