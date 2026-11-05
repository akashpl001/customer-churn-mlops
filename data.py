import numpy as np
import pandas as pd

def generate_data(n=2000, seed=42):
    """Generate a reproducible, illustrative churn dataset."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, n)
    monthly_charges = rng.uniform(25, 120, n).round(2)
    support_calls = rng.poisson(1.4, n)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[.55, .25, .20])
    internet = rng.choice(["fiber", "dsl", "none"], n, p=[.5, .35, .15])
    logit = (-1.7 + .035 * monthly_charges - .045 * tenure
             + .48 * support_calls + 1.0 * (contract == "month-to-month")
             - .45 * (contract == "two-year") + .25 * (internet == "fiber"))
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)
    return pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "support_calls": support_calls,
        "contract_type": contract,
        "internet_service": internet,
        "churn": churn
    })
