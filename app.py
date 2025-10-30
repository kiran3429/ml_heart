import streamlit as st
import pandas as pd
import joblib
import requests
import io

FILE_ID = "1EB9x5IAeSjCx9UWfUqvYZOCHwV8jo-Ty"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def load_model():
    response = requests.get(URL)
    if response.status_code != 200:
        st.error("⚠️ Unable to load model from Google Drive.")
        return None
    model = joblib.load(io.BytesIO(response.content))
    return model

rf_pipeline = load_model()
if rf_pipeline is None:
    st.stop()

st.title("💓 Heart Disease Prediction App")
st.write("Predict the likelihood of heart disease using clinical parameters.")

st.sidebar.header("Enter Patient Details")

age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
chest_pain_type = st.sidebar.selectbox("Chest Pain Type", ["Typical Angina (1)", "Atypical Angina (2)", "Non-Anginal Pain (3)", "Asymptomatic (4)"])
resting_bp_s = st.sidebar.slider("Resting BP (mmHg)", 80, 200, 120)
cholesterol = st.sidebar.slider("Cholesterol (mg/dL)", 100, 400, 200)
fasting_blood_sugar = st.sidebar.selectbox("Fasting Blood Sugar", ["<=120 mg/dL (0)", ">120 mg/dL (1)"])
resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal (0)", "ST-T Abnormality (1)", "LV Hypertrophy (2)"])
max_heart_rate = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.sidebar.selectbox("Exercise Angina", ["No (0)", "Yes (1)"])
oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.5, 1.0, 0.1)
st_slope = st.sidebar.selectbox("ST Slope", ["Upward (1)", "Flat (2)", "Downward (3)"])

input_data = {
    "age": [age],
    "sex": [1 if sex == "Male" else 0],
    "chest_pain_type": [int(chest_pain_type.split("(")[-1][0])],
    "resting_bp_s": [resting_bp_s],
    "cholesterol": [cholesterol],
    "fasting_blood_sugar": [int(fasting_blood_sugar.split("(")[-1][0])],
    "resting_ecg": [int(resting_ecg.split("(")[-1][0])],
    "max_heart_rate": [max_heart_rate],
    "exercise_angina": [1 if exercise_angina == "Yes (1)" else 0],
    "oldpeak": [oldpeak],
    "st_slope": [int(st_slope.split("(")[-1][0])]
}

input_df = pd.DataFrame(input_data)

if st.button("🔍 Predict"):
    pred = rf_pipeline.predict(input_df)[0]
    prob = rf_pipeline.predict_proba(input_df)[0, 1]

    if pred == 1:
        st.error(f"⚠️ High Risk of Heart Disease — Probability: {prob:.2f}")
    else:
        st.success(f"✅ No Heart Disease — Probability: {prob:.2f}")

    st.write("### Input Summary:")
    st.dataframe(input_df)

st.write("---")
st.caption("Developed By Kiran Kumar Annad Hari using Streamlit and Random Forest Classifier")
