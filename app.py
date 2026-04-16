import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("Heart Attack Risk Predictor")

# ===== Inputs (same as dataset) =====
age = st.number_input("Age", 18, 100)
sex = st.selectbox("Sex", ["Male", "Female"])
diabetes = st.selectbox("Diabetes", ["No","Yes"])
family_history = st.selectbox("Family History", ["No","Yes "])
smoking = st.selectbox("Smoking", ["No","Yes"])
obesity = st.selectbox("Obesity", ["No","Yes"])
alcohol = st.selectbox("Alcohol Consumption", ["No","Yes"])
exercise = st.number_input("Exercise Hours Per Week", 0.0, 100.0)
diet = st.selectbox("Diet", ["Healthy", "Average", "Unhealthy"])
prev_heart = st.selectbox("Previous Heart Problems", ["No","Yes"])
medication = st.selectbox("Medication Use", ["No","Yes"])
stress = st.number_input("Stress Level", 0, 10)
sedentary = st.number_input("Sedentary Hours Per Day", 0.0, 15.0)
bmi = st.number_input("BMI", 10.0, 50.0)
physical = st.number_input("Physical Activity Days Per Week", 0, 7)
sleep = st.number_input("Sleep Hours Per Day", 0, 12)

# ===== Predict =====
if st.button("Predict"):

    user_data = {
        "Age": age,
        "Sex": sex,
        "Diabetes": diabetes,
        "Family History": family_history,
        "Smoking": smoking,
        "Obesity": obesity,
        "Alcohol Consumption": alcohol,
        "Exercise Hours Per Week": exercise,
        "Diet": diet,
        "Previous Heart Problems": prev_heart,
        "Medication Use": medication,
        "Stress Level": stress,
        "Sedentary Hours Per Day": sedentary,
        "BMI": bmi,
        "Physical Activity Days Per Week": physical,
        "Sleep Hours Per Day": sleep
    }

    df = pd.DataFrame([user_data])

    # Same preprocessing as notebook
    df = pd.get_dummies(df)

    # Align with training columns (IMPORTANT)
    model_columns = model.feature_names_in_
    df = df.reindex(columns=model_columns, fill_value=0)

    prob = model.predict_proba(df)[0][1]

    st.write(f"Risk Probability: {prob*100:.2f}%")