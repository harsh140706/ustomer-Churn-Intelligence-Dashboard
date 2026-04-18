import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(df: pd.DataFrame):
    """
    Clean and encode the Telco churn dataset.
    Returns X (numpy), y (numpy), feature_names (list).
    """
    data = df.copy()

    # Fix TotalCharges (has blank strings)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"].fillna(data["TotalCharges"].median(), inplace=True)

    # Drop customerID — not a feature
    if "customerID" in data.columns:
        data.drop("customerID", axis=1, inplace=True)

    # Target encoding
    data["Churn"] = (data["Churn"] == "Yes").astype(int)

    # ── Feature Engineering ──────────────────────────────────────────────
    # Revenue per month of tenure (loyalty-adjusted value)
    data["ChargesPerTenureMonth"] = data["MonthlyCharges"] / (data["tenure"] + 1)

    # Charge inflation: how much more are they paying vs what they started with
    data["ChargeInflation"] = data["TotalCharges"] / ((data["tenure"] + 1) * data["MonthlyCharges"] + 1)

    # Count of services subscribed — highly predictive
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in service_cols:
        if col in data.columns:
            data[col + "_bin"] = data[col].apply(
                lambda x: 1 if x == "Yes" else 0
            )
    bin_cols = [c + "_bin" for c in service_cols if c + "_bin" in data.columns]
    data["NumServices"] = data[bin_cols].sum(axis=1)

    # ── Encode categorical columns ───────────────────────────────────────
    cat_cols = data.select_dtypes(include=["object"]).columns.tolist()

    le = LabelEncoder()
    for col in cat_cols:
        data[col] = le.fit_transform(data[col].astype(str))

    # ── Split X / y ──────────────────────────────────────────────────────
    y = data["Churn"].values
    X = data.drop("Churn", axis=1)
    feature_names = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_names
