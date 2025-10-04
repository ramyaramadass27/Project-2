# -------------------------
# streamlit_app.py
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------
# Feature Engineer
# -------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        if 'Annual Income' in X.columns and 'Number of Dependents' in X.columns:
            X['Income_per_Dependent'] = X['Annual Income'] / (X['Number of Dependents'] + 1)
        if 'Age' in X.columns and 'Health Score' in X.columns:
            X['Age_Health_Interaction'] = X['Age'] * X['Health Score']
        return X

# -------------------------
# Load trained pipeline
# -------------------------
model = joblib.load("best_xgb_pipeline1.pkl")

# -------------------------
# Streamlit UI
# -------------------------
st.title("💡 Smart Insurance Premium Predictor")
st.markdown("Enter customer details to predict the insurance premium.")

# --- Numeric Inputs ---
age = st.number_input("Age", min_value=18, max_value=100, value=30)
annual_income = st.number_input("Annual Income", min_value=10000, max_value=1000000, value=50000)
num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=4, value=0)
health_score = st.slider("Health Score (0–60)", 0, 60, 30)
previous_claims = st.number_input("Previous Claims", min_value=0, max_value=9, value=0)
vehicle_age = st.number_input("Vehicle Age", min_value=0, max_value=20, value=1)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
insurance_duration = st.number_input("Insurance Duration (years)", min_value=1, max_value=20, value=5)
policy_start_date = st.date_input("Policy Start Date", min_value=date(2000,1,1), max_value=date.today())

# --- Categorical Inputs ---
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])
location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
exercise_frequency = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo"])

# -------------------------
# Prepare input DataFrame
# -------------------------
input_data = pd.DataFrame({
    "id": [0],  # dummy id to match training columns
    "Age": [age],
    "Annual Income": [annual_income],
    "Number of Dependents": [num_dependents],
    "Health Score": [health_score],
    "Previous Claims": [previous_claims],
    "Vehicle Age": [vehicle_age],
    "Credit Score": [credit_score],
    "Insurance Duration": [insurance_duration],
    "Policy Start Date": [policy_start_date],
    "Gender": [gender],
    "Marital Status": [marital_status],
    "Education Level": [education_level],
    "Occupation": [occupation],
    "Location": [location],
    "Policy Type": [policy_type],
    "Smoking Status": [smoking_status],
    "Exercise Frequency": [exercise_frequency],
    "Property Type": [property_type]
})

# -------------------------
# Predict Button
# -------------------------
if st.button("Predict Premium"):
    try:
        prediction = model.predict(input_data)
        st.success(f"💰 Predicted Insurance Premium: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
