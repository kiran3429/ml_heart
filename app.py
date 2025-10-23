import streamlit as st
import pandas as pd
import numpy as np
import requests
import onnxruntime as rt

# --- Google Drive ONNX file ---
FILE_ID = "1g7MKWjoPey_70xmobQFqGVTcVae1UB3_"  # Replace with your ONNX file ID
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

@st.cache_resource
def load_model():
    model_path = "heart_rf_model.onnx"
    
    # Download ONNX file
    r = requests.get(URL)
    if r.status_code != 200:
        st.error("❌ Failed to download model. Make sure file is shared as 'Anyone with link'.")
        st.stop()
    
    with open(model_path, "wb") as f:
        f.write(r.content)
    
    # Load ONNX model
    return rt.InferenceSession(model_path)

rf_session = load_model()
input_name = rf_session.get_inputs()[0].name

# --- UI ---
st.title("❤️ Heart Disease Prediction System")

age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Female", "Male"])
chestpain = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
)
restingbps = st.slider("Resting BP (mmHg)", 80, 200, 120)
cholesterol = st.slider("Cholesterol (mg/dL)", 100, 400, 200)
fastingbs = st.selectbox("Fasting Blood Sugar", ["<=120 mg/dL", ">120 mg/dL"])
restingecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST-T Abnormality", "LV Hypertrophy"]
)
maxheartrate = st.slider("Max Heart Rate", 60, 220, 150)
exerciseangina = st.selectbox("Exercise Angina", ["No", "Yes"])
oldpeak = st.slider("Oldpeak", 0.0, 6.5, 1.0, 0.1)
STslope = st.selectbox("ST Slope", ["Upward", "Flat", "Downward"])

# --- Convert to numeric like training data ---
sex = 1 if sex == "Male" else 0
chestpain_map = {"Typical Angina": 1, "Atypical Angina": 2, "Non-Anginal Pain": 3, "Asymptomatic": 4}
restingecg_map = {"Normal": 0, "ST-T Abnormality": 1, "LV Hypertrophy": 2}
exerciseangina_map = {"No": 0, "Yes": 1}
fastingbs_map = {"<=120 mg/dL": 0, ">120 mg/dL": 1}
STslope_map = {"Upward": 1, "Flat": 2, "Downward": 3}

input_dict = {
    "age": [age],
    "sex": [sex],
    "chest_pain_type": [chestpain_map[chestpain]],
    "resting_bp_s": [restingbps],
    "cholesterol": [cholesterol],
    "fasting_blood_sugar": [fastingbs_map[fastingbs]],
    "resting_ecg": [restingecg_map[restingecg]],
    "max_heart_rate": [maxheartrate],
    "exercise_angina": [exerciseangina_map[exerciseangina]],
    "oldpeak": [oldpeak],
    "st_slope": [STslope_map[STslope]]
}

input_df = pd.DataFrame(input_dict, dtype=np.float32)

# --- Prediction ---
if st.button("🔍 Predict"):
    pred = rf_session.run(None, {input_name: input_df.values})[0][0]
    if pred == 1:
        st.error("🚨 Heart Disease Detected! Please consult a doctor.")
    else:
        st.success("✅ Normal - No signs of heart disease detected.")
    
    st.write("---")
    st.write("### Input Summary:")
    st.write(input_df)
