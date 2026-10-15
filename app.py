import streamlit as st
import pandas as pd
import numpy as np
import joblib


heart_model = joblib.load('heart_failure_model.pkl')
heart_scaler = joblib.load('scaler.pkl')

st.title("Heart Failure Prediction App")

#age = st.number_input("Age", min_value=1, max_value=120, value=30)
age = st.slider("Age", min_value=40, max_value=100, value=60)
anemia = st.selectbox("Anemia (0: No, 1: Yes)", [0, 1])
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", min_value=0, max_value=10000, value=100) 
diabetes = st.selectbox("Diabetes (0: No, 1: Yes)", [0, 1]) 
ef = st.number_input("Ejection Fraction (%)", min_value=10, max_value=80, value=30)
hbp = st.selectbox("High Blood Pressure (0: No, 1: Yes)", [0, 1])
platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=0, value=250000)
serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, value=1.0)
serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=100,  value=135)
gender = st.selectbox("Gender (0: Female, 1: Male)", [0, 1])
smoking = st.selectbox("Smoking (0: No, 1: Yes)", [0, 1])
time = st.number_input("Follow-up Period (days)", min_value=0, value=100)

if st.button("Predict"):
    heart_input = ([[age, anemia, creatinine_phosphokinase, diabetes, ef, hbp, platelets, serum_creatinine, serum_sodium, gender, smoking, time]])
    scaled = heart_scaler.transform(heart_input)
    prediction = heart_model.predict(scaled)
    if prediction[0] == 1:
        st.error("The patient is at risk of heart failure.")
    else:
        st.success("The patient is not at risk of heart failure.")
    
    