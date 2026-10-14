import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

model = joblib.load("churn_model.pkl")
meta = json.load(open("model_meta.json"))
COLUMNS, THRESHOLD = meta["columns"], meta["threshold"]

SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def build_features(raw: dict) -> pd.DataFrame:
    """Repeat the exact feature engineering used in the notebook."""
    d = pd.DataFrame([raw])
    d["NumServices"] = (d[SERVICE_COLS] == "Yes").sum(axis=1)
    d["TenureGroup"] = pd.cut(d["tenure"], bins=[-1, 12, 24, 48, 72],
                              labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)
    d["AutoPay"] = d["PaymentMethod"].str.contains("automatic").astype(int)
    # no drop_first here: a single row would lose its category. reindex drops the baseline columns.
    d = pd.get_dummies(d, dtype=int)
    return d.reindex(columns=COLUMNS, fill_value=0)


st.title("📉 Customer Churn Predictor")
st.caption("Logistic Regression trained on the Telco Customer Churn dataset. Enter a customer's details to estimate churn risk.")

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Account")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                              "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    monthly = st.number_input("Monthly charges", 18.0, 120.0, 70.0, step=0.5)
with c2:
    st.subheader("Customer")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
with c3:
    st.subheader("Services")
    phone = st.selectbox("Phone service", ["Yes", "No"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    no_net = internet == "No"
    yn = ["No internet service"] if no_net else ["Yes", "No"]
    multi = st.selectbox("Multiple lines", ["No phone service"] if phone == "No" else ["Yes", "No"])
    security = st.selectbox("Online security", yn)
    backup = st.selectbox("Online backup", yn)
    protection = st.selectbox("Device protection", yn)
    support = st.selectbox("Tech support", yn)
    tv = st.selectbox("Streaming TV", yn)
    movies = st.selectbox("Streaming movies", yn)

if st.button("Predict churn risk", type="primary"):
    raw = {"gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
           "tenure": tenure, "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
           "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
           "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
           "PaperlessBilling": paperless, "PaymentMethod": payment, "MonthlyCharges": monthly}
    prob = float(model.predict_proba(build_features(raw))[:, 1][0])
    segment = "Low" if prob < 0.3 else "Medium" if prob < 0.6 else "High"

    m1, m2, m3 = st.columns(3)
    m1.metric("Churn probability", f"{prob:.1%}")
    m2.metric("Risk segment", segment)
    m3.metric("Prediction", "Likely to churn" if prob >= THRESHOLD else "Likely to stay")
    st.progress(min(prob, 1.0))

    tips = []
    if contract == "Month-to-month":
        tips.append("Offer a discount to move this customer to a one- or two-year contract.")
    if payment == "Electronic check":
        tips.append("Encourage automatic payment with a small incentive.")
    if internet == "Fiber optic":
        tips.append("Check fiber service quality and price competitiveness for this customer.")
    if tenure <= 12:
        tips.append("New customer: a proactive onboarding call can reduce early churn.")
    if tips:
        st.subheader("Suggested retention actions")
        for t in tips:
            st.write("•", t)
