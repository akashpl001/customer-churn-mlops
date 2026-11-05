# Customer Churn Prediction | End-to-End MLOps

A portfolio-ready Data Science project demonstrating model training, API serving, Docker, GitHub Actions CI/CD, and Azure deployment.

> **Dataset note:** This starter project generates synthetic customer data for reproducibility. Replace it with a properly licensed real churn dataset before making business claims. Metrics from synthetic data are not evidence of real-world performance.

## Architecture

```text
Synthetic / real customer data
          ↓
Pandas + Scikit-learn training pipeline
          ↓
Serialized model (joblib)
          ↓
FastAPI prediction endpoint
          ↓
Docker image
          ↓
GitHub Actions: train → test → build
          ↓
Azure Container Registry → Azure Container Apps
```

## Run locally

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python -m src.train
uvicorn src.api:app --reload
```

Open:
- API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## Example prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 4,
    "monthly_charges": 95,
    "support_calls": 3,
    "contract_type": "month-to-month",
    "internet_service": "fiber"
  }'
```

## Run tests

```bash
python -m src.train
pytest -q
```

## Docker

```bash
docker build -t customer-churn-api .
docker run --rm -p 8000:8000 customer-churn-api
```

Then visit http://localhost:8000/docs.

## GitHub Actions

`ci.yml` runs on pushes and pull requests to `main`:
1. Installs Python dependencies
2. Trains the model
3. Runs API tests
4. Builds the Docker image

`deploy-azure.yml` deploys to Azure Container Apps when pushed to `main` (or manually triggered). Configure these GitHub repository secrets:

| Secret | Value |
|---|---|
| `AZURE_CREDENTIALS` | Azure service principal JSON |
| `AZURE_RESOURCE_GROUP` | Resource group name |
| `AZURE_ACR_NAME` | Azure Container Registry name (name only, no `.azurecr.io`) |
| `AZURE_CONTAINERAPP_NAME` | Existing Container App name |

The Azure Container App must already exist and be configured to pull from your ACR. Grant the app's managed identity `AcrPull` on the registry, or configure registry credentials. The service principal used by GitHub Actions needs permission to build/push to ACR and update the Container App.

## Azure setup outline

1. Create an Azure resource group.
2. Create an Azure Container Registry (ACR).
3. Create an Azure Container App and Container Apps environment.
4. Configure ACR image pull access for the Container App.
5. Add the four GitHub Actions secrets listed above.
6. Push to `main`; monitor the Actions tab.
7. Open the Container App's HTTPS ingress URL and test `/docs`.

Azure resources may incur charges. Review pricing, quotas, and resource cleanup before deployment.

## Project structure

```text
customer-churn-mlops/
├── src/
│   ├── data.py
│   ├── train.py
│   └── api.py
├── tests/
│   └── test_api.py
├── .github/workflows/
│   ├── ci.yml
│   └── deploy-azure.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Next improvements

- Replace synthetic data with a public/licensed dataset.
- Add cross-validation, hyperparameter tuning, and model explainability (SHAP).
- Track experiments and model versions (MLflow).
- Add monitoring for prediction drift and API latency.
- Add a dashboard (Power BI or Streamlit) for churn insights.
- Add security scanning and Azure Key Vault integration.

## Resume bullet (after you complete and validate deployment)

Built and deployed an end-to-end customer churn prediction API using Python, Scikit-learn, FastAPI, and Docker; automated model testing and container builds with GitHub Actions, and deployed the containerized service to Azure Container Apps.
