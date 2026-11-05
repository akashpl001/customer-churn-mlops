from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction_endpoint():
    payload = {
        "tenure_months": 4,
        "monthly_charges": 95.0,
        "support_calls": 3,
        "contract_type": "month-to-month",
        "internet_service": "fiber"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert 0 <= result["churn_probability"] <= 1
    assert result["prediction"] in ["likely_to_churn", "unlikely_to_churn"]
