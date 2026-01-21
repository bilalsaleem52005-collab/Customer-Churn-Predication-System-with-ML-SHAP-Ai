import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📉",
    layout="wide"
)

st.title("📊 Customer Churn Prediction System")
st.markdown(
    "This system predicts whether a customer is likely to **churn** and provides **actionable recommendations**."
)

# -----------------------------
# LOAD MODEL (SAFE PATH)
# -----------------------------
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_churn_model.pkl")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("🧾 Customer Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents?", ["No", "Yes"])

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 20, 120, 70)
total_charges = st.sidebar.slider("Total Charges ($)", 0, 9000, 1000)

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
payment = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def build_input():
    data = {
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "gender_Male": 1 if gender == "Male" else 0,
        "Partner_Yes": 1 if partner == "Yes" else 0,
        "Dependents_Yes": 1 if dependents == "Yes" else 0,
        "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
        "InternetService_No": 1 if internet == "No" else 0,
        "TechSupport_Yes": 1 if tech_support == "Yes" else 0,
        "OnlineSecurity_Yes": 1 if online_security == "Yes" else 0,
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,
        "PaymentMethod_Electronic check": 1 if payment == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if payment == "Mailed check" else 0,
    }

    return pd.DataFrame([data])

# -----------------------------
# PREDICTION
# -----------------------------
st.markdown("---")
st.subheader("🔍 Prediction Result")

if st.button("🚀 Predict Churn"):
    input_df = build_input()

    # Align columns with training model
    for col in model.get_booster().feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.get_booster().feature_names]

    probability = model.predict_proba(input_df)[0][1]
    prediction = "YES" if probability >= 0.5 else "NO"

    col1, col2 = st.columns(2)

    with col1:
        if prediction == "YES":
            st.error(f"⚠️ Customer is likely to CHURN")
        else:
            st.success(f"✅ Customer is likely to STAY")

    with col2:
        st.metric("Churn Probability", f"{probability:.2%}")

    # -----------------------------
    # RECOMMENDATIONS
    # -----------------------------
    st.markdown("---")
    st.subheader("💡 Recommended Retention Actions")

    recommendations = []

    if contract == "Month-to-month":
        recommendations.append("📄 Offer a discounted long-term contract")

    if monthly_charges > 80:
        recommendations.append("💸 Provide a temporary price reduction or loyalty discount")

    if tech_support == "No":
        recommendations.append("🛠️ Offer free tech support for 3 months")

    if online_security == "No":
        recommendations.append("🔐 Bundle free online security service")

    if tenure < 12:
        recommendations.append("🎁 Give onboarding rewards for new customers")

    if not recommendations:
        recommendations.append("👍 Maintain current service quality and engagement")

    for r in recommendations:
        st.write("•", r)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Final Year Project | AI-Based Customer Churn Prediction System")
