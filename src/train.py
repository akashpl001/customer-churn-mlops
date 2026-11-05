from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from src.data import generate_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    df = generate_data()
    X, y = df.drop(columns="churn"), df["churn"]
    numeric = ["tenure_months", "monthly_charges", "support_calls"]
    categorical = ["contract_type", "internet_service"]
    prep = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])
    model = Pipeline([
        ("preprocess", prep),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, probabilities):.3f}")
    print(classification_report(y_test, predictions))
    joblib.dump(model, MODEL_DIR / "churn_pipeline.joblib")
    print("Saved model to models/churn_pipeline.joblib")

if __name__ == "__main__":
    main()
