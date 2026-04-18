import numpy as np
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV


def train_model(X, y, test_size=0.2, random_state=42):
    """
    Train an XGBoost model with class imbalance handling.
    Returns fitted model, X_test, y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Handle class imbalance
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    return model, X_test, y_test


def get_shap_values(model, X_test, feature_names, n_sample=300):
    """
    Compute SHAP values using TreeExplainer (fast for XGBoost).
    Returns (shap_values array, explainer, X_sample).
    """
    # Sample for speed in Streamlit
    n = min(n_sample, len(X_test))
    idx = np.random.choice(len(X_test), n, replace=False)
    X_sample = X_test[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification XGBoost, shap_values may be list — take class-1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values, explainer, X_sample
