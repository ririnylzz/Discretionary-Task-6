from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Inisialisasi FastAPI
app = FastAPI(title="Pollution Prediction API")

# Load model dan scaler dari file polusi.pkl
with open("polusi.pkl", "rb") as f:
    saved_objects = pickle.load(f)
    model = saved_objects['model']
    scaler = saved_objects['scaler']

# Skema input sesuai dengan dataset pelatihan
class InputData(BaseModel):
    PM10: float
    SO2: float
    CO: float
    O3: float
    NO2: float
    Max: float
    Pollution_Average: float
    Month: int
    Weekday: int
    Quarter: int

# Preprocessing input
def preprocess_input(data: InputData):
    # Membuat DataFrame sesuai dengan kolom yang ada di data pelatihan
    df = pd.DataFrame([{
        "PM10": data.PM10,
        "SO2": data.SO2,
        "CO": data.CO,
        "O3": data.O3,
        "NO2": data.NO2,
        "Max": data.Max,
        "Pollution_Average": data.Pollution_Average,
        "Month": data.Month,
        "Weekday": data.Weekday,
        "Quarter": data.Quarter
    }])
    # Melakukan transformasi data (scaling)
    df_scaled = scaler.transform(df)
    return df_scaled

@app.get("/")
def read_root():
    return {"message": "Pollution Prediction API is running"}

# Endpoint prediksi
@app.post("/predict")
def predict_pollution(data: InputData):
    processed = preprocess_input(data)
    prediction = model.predict(processed)[0]
    return {
        "predicted_value": round(float(prediction), 2)
    }
