import numpy as np


def compute_business_impact(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    cac: float,
    arpu: float,
    retention_cost: float,
    retention_success_rate: float,
    threshold: float = 0.5,
    top_pct: float = 0.20
) -> dict:
    """
    Translate model predictions into business ROI metrics.

    Parameters
    ----------
    y_prob               : Predicted churn probabilities
    y_true               : Actual churn labels (0/1)
    cac                  : Customer acquisition cost ($)
    arpu                 : Average revenue per user per month ($)
    retention_cost       : Cost to reach out to one at-risk customer ($)
    retention_success_rate: Fraction of reached customers who stay
    threshold            : Decision boundary for flagging as churner
    top_pct              : Fraction of highest-risk customers to target

    Returns
    -------
    dict with calculated impact metrics
    """
    predicted_churners = (y_prob >= threshold).sum()
    revenue_at_risk = predicted_churners * arpu

    # Target only the top % most at-risk (budget-constrained strategy)
    n_target = max(1, int(len(y_prob) * top_pct))
    top_idx = np.argsort(y_prob)[::-1][:n_target]

    outreach_cost = n_target * retention_cost
    customers_saved = int(n_target * retention_success_rate)

    # Each saved customer = 12 months ARPU recovered (annual LTV)
    revenue_saved = customers_saved * arpu * 12
    # Minus what it cost us to reach them
    net_saved = revenue_saved - outreach_cost

    # CAC equivalent: if we didn't retain, we'd need to re-acquire
    cac_equivalent = customers_saved * cac

    # Precision at top decile: what fraction of targeted are true churners
    if len(top_idx) > 0:
        precision_at_top = y_true[top_idx].mean()
    else:
        precision_at_top = 0.0

    return {
        "n_predicted_churn": int(predicted_churners),
        "revenue_at_risk": float(revenue_at_risk),
        "n_targeted": n_target,
        "outreach_cost": float(outreach_cost),
        "customers_saved": customers_saved,
        "revenue_saved": float(revenue_saved),
        "net_saved": float(net_saved),
        "cac_equivalent": float(cac_equivalent),
        "precision_at_top": float(precision_at_top),
    }
