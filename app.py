import streamlit as st
import pickle
import pandas as pd

# -------------------------------
# Load saved files
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Churn Prediction", page_icon="📊")

st.title("📊 Customer Churn Prediction App")
st.write("Predict whether a customer will churn or not.")

# -------------------------------
# User Inputs
# -------------------------------

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# -------------------------------
# Preprocessing Function
# -------------------------------
def preprocess_input(user_input_dict):
    input_df = pd.DataFrame([user_input_dict])

    # Apply same encoding
    input_df = pd.get_dummies(input_df)

    # Align with training columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    return input_df

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict"):

    user_input = {
        'tenure': tenure,
        'monthlycharges': monthly_charges,
        'totalcharges': total_charges,
        'contract': contract,
        'internetservice': internet,
        'techsupport': tech_support,
        'onlinesecurity': online_security,
        'onlinebackup': online_backup,
        'deviceprotection': device_protection,
        'streamingtv': streaming_tv,
        'streamingmovies': streaming_movies
    }

    input_df = preprocess_input(user_input)
    st.write("Input DF Columns:", input_df.columns.tolist())
    st.write("Training Columns:", columns.tolist())
    # Scale input
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # -------------------------------
    # Output
    # -------------------------------
    st.subheader("Result:")

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Customer is likely to stay\n\nProbability: {probability:.2f}")

