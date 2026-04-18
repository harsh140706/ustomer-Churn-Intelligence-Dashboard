import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import joblib
import os
from src.preprocess import preprocess_data
from src.model import train_model, get_shap_values
from src.insights import compute_business_impact

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #e74c3c;
    }
    .metric-label { font-size: 13px; color: #888; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: 700; color: #1a1a2e; }
    .metric-delta { font-size: 12px; color: #27ae60; }
    .section-title {
        font-size: 18px; font-weight: 600;
        border-bottom: 2px solid #e74c3c;
        padding-bottom: 6px; margin: 1.5rem 0 1rem;
    }
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #eee;
        border-radius: 10px;
        padding: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/combo-chart.png", width=60)
    st.title("Churn Predictor")
    st.caption("Telco Customer Analytics")
    st.divider()

    uploaded = st.file_uploader("Upload Customer CSV", type=["csv"])
    st.caption("Or use the default Telco dataset")
    st.divider()

    st.subheader("Filters")
    contract_filter = st.multiselect(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"],
        default=["Month-to-month", "One year", "Two year"]
    )
    tenure_range = st.slider("Tenure (months)", 0, 72, (0, 72))
    charge_range = st.slider("Monthly Charges ($)", 0, 120, (0, 120))

# ── Load & preprocess data ───────────────────────────────────────────────────
@st.cache_data
def load_data(file=None):
    if file:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv("data/telco_churn.csv")
    return df

@st.cache_resource
def get_trained_model(df):
    X, y, feature_names = preprocess_data(df)
    model, X_test, y_test = train_model(X, y)
    return model, X_test, y_test, feature_names

try:
    raw_df = load_data(uploaded)
    model, X_test, y_test, feature_names = get_trained_model(raw_df)
    data_loaded = True
except FileNotFoundError:
    st.error("⚠️ Please upload a CSV file or place `telco_churn.csv` in the `data/` folder.")
    st.info("Download from: https://www.kaggle.com/blastchar/telco-customer-churn")
    data_loaded = False
    st.stop()

# Apply sidebar filters
df = raw_df.copy()
df = df[df["Contract"].isin(contract_filter)]
df = df[df["tenure"].between(*tenure_range)]
df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
df = df[df["MonthlyCharges"].between(*charge_range)]

total = len(df)
churned = df["Churn"].value_counts().get("Yes", 0)
churn_rate = churned / total if total else 0
avg_charges = df["MonthlyCharges"].mean()
avg_tenure = df["tenure"].mean()

# ── Header ──────────────────────────────────────────────────────────────────
st.title("📉 Customer Churn Intelligence Dashboard")
st.caption(f"Showing {total:,} customers · Filtered view")

# ── KPI Row ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Customers", f"{total:,}")
k2.metric("Churned", f"{churned:,}", delta=f"{churn_rate:.1%} churn rate", delta_color="inverse")
k3.metric("Avg Monthly Charges", f"${avg_charges:.0f}")
k4.metric("Avg Tenure", f"{avg_tenure:.0f} months")

st.divider()

# ── Tab Layout ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA & Overview", "🤖 Model Performance", "🔍 SHAP Explainability", "💼 Business Impact", "🧍 Predict Single Customer"
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<p class="section-title">Churn Distribution</p>', unsafe_allow_html=True)
        churn_counts = df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Status", "Count"]
        fig = px.pie(
            churn_counts, values="Count", names="Status",
            color="Status",
            color_discrete_map={"Yes": "#e74c3c", "No": "#2ecc71"},
            hole=0.55
        )
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(margin=dict(t=10, b=10), height=280, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<p class="section-title">Churn by Contract Type</p>', unsafe_allow_html=True)
        contract_churn = df.groupby(["Contract", "Churn"]).size().reset_index(name="Count")
        fig2 = px.bar(
            contract_churn, x="Contract", y="Count", color="Churn",
            barmode="group", color_discrete_map={"Yes": "#e74c3c", "No": "#3498db"}
        )
        fig2.update_layout(margin=dict(t=10, b=10), height=280, legend_title="Churned")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<p class="section-title">Tenure vs Monthly Charges</p>', unsafe_allow_html=True)
        sample = df.sample(min(800, len(df)), random_state=42)
        fig3 = px.scatter(
            sample, x="tenure", y="MonthlyCharges", color="Churn",
            color_discrete_map={"Yes": "#e74c3c", "No": "#95a5a6"},
            opacity=0.6, size_max=6
        )
        fig3.update_layout(margin=dict(t=10, b=10), height=280)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown('<p class="section-title">Monthly Charges Distribution</p>', unsafe_allow_html=True)
        fig4 = px.histogram(
            df, x="MonthlyCharges", color="Churn", nbins=40,
            color_discrete_map={"Yes": "#e74c3c", "No": "#3498db"},
            barmode="overlay", opacity=0.7
        )
        fig4.update_layout(margin=dict(t=10, b=10), height=280)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<p class="section-title">Churn Rate by Service Add-ons</p>', unsafe_allow_html=True)
    service_cols = ["PhoneService", "InternetService", "TechSupport", "StreamingTV", "OnlineSecurity"]
    service_churn = []
    for col in service_cols:
        if col in df.columns:
            grp = df[df[col] != "No internet service"].groupby(col)["Churn"].apply(
                lambda x: (x == "Yes").mean()
            ).reset_index()
            grp.columns = [col, "Churn Rate"]
            grp["Service"] = col
            grp.rename(columns={col: "Value"}, inplace=True)
            service_churn.append(grp)

    if service_churn:
        sc_df = pd.concat(service_churn)
        fig5 = px.bar(sc_df, x="Value", y="Churn Rate", facet_col="Service",
                      facet_col_wrap=3, color="Churn Rate",
                      color_continuous_scale=["#2ecc71", "#e74c3c"])
        fig5.update_layout(height=400, margin=dict(t=40, b=10))
        st.plotly_chart(fig5, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_curve, auc, precision_recall_curve
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{report['accuracy']:.1%}")
    m2.metric("Precision (Churn)", f"{report['1']['precision']:.1%}")
    m3.metric("Recall (Churn)", f"{report['1']['recall']:.1%}")
    m4.metric("ROC-AUC", f"{roc_auc:.3f}")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<p class="section-title">Confusion Matrix</p>', unsafe_allow_html=True)
        labels = ["Not Churned", "Churned"]
        fig_cm = px.imshow(
            cm, text_auto=True, x=labels, y=labels,
            color_continuous_scale=["#eaf4fb", "#2980b9"],
            labels=dict(x="Predicted", y="Actual")
        )
        fig_cm.update_layout(height=300, margin=dict(t=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    with c2:
        st.markdown('<p class="section-title">ROC Curve</p>', unsafe_allow_html=True)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"XGBoost (AUC={roc_auc:.3f})",
            line=dict(color="#e74c3c", width=2.5)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random",
            line=dict(dash="dash", color="#aaa", width=1)
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=300, margin=dict(t=10),
            legend=dict(x=0.6, y=0.1)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown('<p class="section-title">Churn Probability Distribution</p>', unsafe_allow_html=True)
    prob_df = pd.DataFrame({"Probability": y_prob, "Actual": ["Churned" if v else "Retained" for v in y_test]})
    fig_prob = px.histogram(
        prob_df, x="Probability", color="Actual", nbins=50,
        color_discrete_map={"Churned": "#e74c3c", "Retained": "#3498db"},
        barmode="overlay", opacity=0.75
    )
    fig_prob.add_vline(x=0.5, line_dash="dash", line_color="orange", annotation_text="Decision Threshold")
    fig_prob.update_layout(height=280, margin=dict(t=10))
    st.plotly_chart(fig_prob, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("SHAP (SHapley Additive exPlanations) shows **why** the model made each prediction — not just what.")

    shap_values, shap_explainer, X_sample = get_shap_values(model, X_test, feature_names)

    st.markdown('<p class="section-title">Global Feature Importance (SHAP)</p>', unsafe_allow_html=True)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean |SHAP|": mean_abs_shap
    }).sort_values("Mean |SHAP|", ascending=True).tail(15)

    fig_shap = px.bar(
        shap_df, x="Mean |SHAP|", y="Feature",
        orientation="h", color="Mean |SHAP|",
        color_continuous_scale=["#fadbd8", "#e74c3c"]
    )
    fig_shap.update_layout(height=420, margin=dict(t=10), showlegend=False,
                            coloraxis_showscale=False)
    st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown('<p class="section-title">Individual Customer Explanation</p>', unsafe_allow_html=True)
    customer_idx = st.slider("Select a customer (test set index)", 0, len(X_sample) - 1, 0)

    ind_shap = shap_values[customer_idx]
    ind_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": ind_shap
    }).sort_values("SHAP Value")

    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in ind_df["SHAP Value"]]
    fig_ind = go.Figure(go.Bar(
        x=ind_df["SHAP Value"], y=ind_df["Feature"],
        orientation="h", marker_color=colors
    ))
    fig_ind.add_vline(x=0, line_color="#333", line_width=1)
    fig_ind.update_layout(
        height=400, margin=dict(t=10),
        xaxis_title="SHAP Value (red = pushes toward churn, green = pushes toward retention)"
    )
    st.plotly_chart(fig_ind, use_container_width=True)

    pred_prob = model.predict_proba(X_sample[customer_idx:customer_idx+1])[0][1]
    churn_label = "🔴 HIGH CHURN RISK" if pred_prob > 0.5 else "🟢 LOW CHURN RISK"
    st.info(f"**Customer #{customer_idx}** — Churn probability: **{pred_prob:.1%}** — {churn_label}")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — BUSINESS IMPACT
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("Translate model predictions into **dollars-and-cents** business decisions.")

    col1, col2 = st.columns(2)
    with col1:
        cac = st.number_input("Customer Acquisition Cost ($)", value=300, step=10)
        arpu = st.number_input("Avg Revenue Per User / month ($)", value=65, step=5)
    with col2:
        retention_cost = st.number_input("Retention Outreach Cost per customer ($)", value=30, step=5)
        retention_rate = st.slider("Assumed Retention Success Rate (%)", 10, 80, 35) / 100

    impact = compute_business_impact(
        y_prob, y_test, cac, arpu, retention_cost, retention_rate, threshold=0.5
    )

    st.markdown('<p class="section-title">Revenue at Risk & ROI</p>', unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Predicted Churners", f"{impact['n_predicted_churn']:,}")
    b2.metric("Revenue at Risk / mo", f"${impact['revenue_at_risk']:,.0f}")
    b3.metric("Outreach Cost (top 20%)", f"${impact['outreach_cost']:,.0f}")
    b4.metric("Net Revenue Saved", f"${impact['net_saved']:,.0f}", delta="with retention campaign")

    st.markdown('<p class="section-title">Optimal Intervention Strategy</p>', unsafe_allow_html=True)
    thresholds = np.arange(0.2, 0.9, 0.05)
    roi_data = []
    for t in thresholds:
        imp = compute_business_impact(y_prob, y_test, cac, arpu, retention_cost, retention_rate, threshold=t)
        roi_data.append({"Threshold": round(t, 2), "Net Saved ($)": imp["net_saved"],
                         "Outreach Cost ($)": imp["outreach_cost"]})
    roi_df = pd.DataFrame(roi_data)

    fig_roi = go.Figure()
    fig_roi.add_trace(go.Scatter(x=roi_df["Threshold"], y=roi_df["Net Saved ($)"],
                                  mode="lines+markers", name="Net Revenue Saved",
                                  line=dict(color="#2ecc71", width=2)))
    fig_roi.add_trace(go.Scatter(x=roi_df["Threshold"], y=roi_df["Outreach Cost ($)"],
                                  mode="lines+markers", name="Outreach Cost",
                                  line=dict(color="#e74c3c", width=2, dash="dash")))
    fig_roi.update_layout(
        xaxis_title="Decision Threshold", yaxis_title="$ (test set, scaled)",
        height=320, margin=dict(t=10), legend=dict(x=0.6, y=0.95)
    )
    st.plotly_chart(fig_roi, use_container_width=True)

    best_thresh = roi_df.loc[roi_df["Net Saved ($)"].idxmax(), "Threshold"]
    st.success(f"**Optimal threshold: {best_thresh}** — Maximizes net revenue saved given your CAC and retention assumptions.")

    st.markdown('<p class="section-title">At-Risk Customer Export</p>', unsafe_allow_html=True)
    proba_full = model.predict_proba(X_test)[:, 1]
    risk_df = pd.DataFrame({
        "Customer Index": range(len(proba_full)),
        "Churn Probability": proba_full.round(3),
        "Risk Tier": pd.cut(proba_full, bins=[0, 0.3, 0.6, 1.0],
                             labels=["Low", "Medium", "High"])
    }).sort_values("Churn Probability", ascending=False)

    st.dataframe(risk_df.head(50), use_container_width=True)
    csv = risk_df.to_csv(index=False).encode()
    st.download_button("⬇ Download At-Risk Customers CSV", csv, "at_risk_customers.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — SINGLE CUSTOMER PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("Fill in a customer's details below to get an **instant churn prediction** with explanation.")

    st.markdown('<p class="section-title">Customer Details</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Account Info**")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        paperless_billing = st.radio("Paperless Billing", ["Yes", "No"], horizontal=True)

    with col2:
        st.markdown("**Charges**")
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0,
                                         value=float(monthly_charges * tenure), step=10.0)
        st.markdown("**Demographics**")
        senior_citizen = st.radio("Senior Citizen", ["No", "Yes"], horizontal=True)
        partner = st.radio("Has Partner", ["Yes", "No"], horizontal=True)
        dependents = st.radio("Has Dependents", ["Yes", "No"], horizontal=True)

    with col3:
        st.markdown("**Services**")
        phone_service = st.radio("Phone Service", ["Yes", "No"], horizontal=True)
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.divider()
    predict_btn = st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True)

    if predict_btn:
        # Build a single-row DataFrame matching the training schema
        customer = pd.DataFrame([{
            "customerID": "SINGLE-PRED",
            "gender": "Male",          # not predictive — kept as neutral default
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": str(total_charges),
            "Churn": "No"  # placeholder — will be dropped in preprocess
        }])

        # Preprocess and predict
        X_single, _, feat_names = preprocess_data(customer)
        prob = model.predict_proba(X_single)[0][1]
        prediction = "Churn" if prob >= 0.5 else "Stay"

        # ── Result card ──────────────────────────────────────────────────
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if prob >= 0.65:
                st.error(f"### 🔴 HIGH RISK\n**Churn Probability: {prob:.1%}**\n\nThis customer is very likely to leave. Immediate outreach recommended.")
            elif prob >= 0.4:
                st.warning(f"### 🟡 MEDIUM RISK\n**Churn Probability: {prob:.1%}**\n\nWatch this customer. Consider a proactive check-in or upgrade offer.")
            else:
                st.success(f"### 🟢 LOW RISK\n**Churn Probability: {prob:.1%}**\n\nThis customer appears stable. No immediate action needed.")

            # Risk gauge using plotly
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number={"suffix": "%", "font": {"size": 28}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#e74c3c" if prob >= 0.5 else "#f39c12" if prob >= 0.35 else "#2ecc71"},
                    "steps": [
                        {"range": [0, 35], "color": "#d5f5e3"},
                        {"range": [35, 65], "color": "#fef9e7"},
                        {"range": [65, 100], "color": "#fadbd8"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 2}, "value": 50}
                }
            ))
            fig_gauge.update_layout(height=220, margin=dict(t=10, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.markdown('<p class="section-title">Why this prediction? (SHAP)</p>', unsafe_allow_html=True)

            # SHAP for single customer
            explainer_single = shap.TreeExplainer(model)
            sv_single = explainer_single.shap_values(X_single)
            if isinstance(sv_single, list):
                sv_single = sv_single[1]

            ind_df = pd.DataFrame({
                "Feature": feat_names,
                "SHAP Value": sv_single[0]
            }).sort_values("SHAP Value").tail(12)

            colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in ind_df["SHAP Value"]]
            fig_shap_single = go.Figure(go.Bar(
                x=ind_df["SHAP Value"], y=ind_df["Feature"],
                orientation="h", marker_color=colors
            ))
            fig_shap_single.add_vline(x=0, line_color="#555", line_width=1)
            fig_shap_single.update_layout(
                height=320, margin=dict(t=10, b=10),
                xaxis_title="← Reduces churn risk | Increases churn risk →"
            )
            st.plotly_chart(fig_shap_single, use_container_width=True)

        # ── Actionable recommendations ────────────────────────────────────
        st.markdown('<p class="section-title">Recommended Actions</p>', unsafe_allow_html=True)
        actions = []
        if contract == "Month-to-month":
            actions.append("📋 **Offer a discounted annual contract** — month-to-month customers churn at 3× the rate of annual customers")
        if tenure < 6:
            actions.append("🤝 **Trigger onboarding check-in** — first 6 months are the highest-risk period; a success call reduces early churn significantly")
        if online_security == "No" and internet_service != "No":
            actions.append("🔒 **Offer a free 1-month Online Security trial** — customers without security add-ons churn at higher rates")
        if tech_support == "No" and internet_service != "No":
            actions.append("🛠 **Suggest TechSupport add-on** — reduces perceived friction and increases switching cost")
        if monthly_charges > 80 and len(actions) == 0:
            actions.append("💰 **Review pricing fit** — high-charge customers with few add-ons perceive low value; consider a plan review call")
        if not actions:
            actions.append("✅ **No immediate action needed** — customer profile is stable. Schedule a routine satisfaction check in 90 days.")

        for action in actions:
            st.markdown(action)