from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_pipeline.joblib"
if not MODEL_PATH.exists():
    raise RuntimeError("Model not found. Run `python -m src.train` first.")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

class Customer(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(gt=0, le=1000)
    support_calls: int = Field(ge=0, le=100)
    contract_type: str
    internet_service: str

@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(customer: Customer):
    row = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict_proba(row)[0, 1])
    return {
        "churn_probability": round(probability, 4),
        "prediction": "likely_to_churn" if probability >= 0.5 else "unlikely_to_churn"
    }
