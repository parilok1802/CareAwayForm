import streamlit as st
import pandas as pd
import joblib

# --- Load trained model ---
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🩺 CareAway Smart Form - AI Assistant")

age = st.number_input("Patient Age", 18, 100, 30)
gender = st.selectbox("Gender", ["M", "F"])
service_type = st.selectbox("Service Type", ["Consultation", "Physiotherapy", "Dentistry"])
symptom = st.selectbox("Symptom", ["Pain", "Injury", "Checkup"])
history = st.radio("Chronic History", [0, 1], index=0)

# Prepare input for prediction
new_patient = pd.DataFrame({
    "age": [age],
    "gender": [gender],
    "service_type": [service_type],
    "symptom": [symptom],
    "history": [history]
})

# One-hot encoding and align with training columns
new_patient_enc = pd.get_dummies(new_patient).reindex(columns=model_columns, fill_value=0)

if st.button("🔮 Predict Next Service"):
    prediction = model.predict(new_patient_enc)
    st.success(f"👉 Suggested Next Service: **{prediction[0]}**")
