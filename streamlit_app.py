"""
=============================================================================
Edge AI Fall Detection System — Streamlit Web App
=============================================================================
Model    : Quantised TFLite Neural Network
Threshold: 0.3 (optimised for Fall Recall = 0.8213)
Input    : 28 extracted accelerometer features OR CSV upload
Dataset  : SisFall (Sucerquia et al., 2017)
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from ai_edge_litert.interpreter import Interpreter

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Edge AI Fall Detection",
    page_icon="🏥",
    layout="wide"
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
THRESHOLD = 0.3

FEATURE_NAMES = []
for axis in ['X', 'Y', 'Z']:
    for stat in ['Mean', 'Std', 'Min', 'Max', 'Range', 'RMS', 'Skewness', 'Kurtosis']:
        FEATURE_NAMES.append(f"{axis}_{stat}")
FEATURE_NAMES += ['SMA', 'Magnitude_Mean', 'Peak_Acceleration', 'Magnitude_Std']

# ─────────────────────────────────────────────
# LOAD MODEL & SCALER (Cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model_and_scaler():
    try:
        interpreter = Interpreter(model_path="model_quantised.tflite")
        interpreter.allocate_tensors()
        scaler = joblib.load("scaler.pkl")
        return interpreter, scaler
    except Exception as e:
        st.error(f"Failed to load model or scaler: {e}")
        st.stop()

interpreter, scaler = load_model_and_scaler()

# ─────────────────────────────────────────────
# INFERENCE FUNCTION
# ─────────────────────────────────────────────
def predict_tflite(interpreter, scaler, features):
    features_scaled = scaler.transform(features.reshape(1, -1)).astype(np.float32)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], features_scaled)
    interpreter.invoke()
    
    probability = interpreter.get_tensor(output_details[0]['index'])[0][0]
    prediction = 1 if probability >= THRESHOLD else 0
    
    return float(probability), prediction

# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_features_from_raw(data):
    x = data['X'].values
    y = data['Y'].values
    z = data['Z'].values
    features = []
    
    for axis in [x, y, z]:
        features.append(np.mean(axis))
        features.append(np.std(axis))
        features.append(np.min(axis))
        features.append(np.max(axis))
        features.append(np.max(axis) - np.min(axis))
        features.append(np.sqrt(np.mean(axis ** 2)))
        
        mean = np.mean(axis)
        std = np.std(axis)
        if std > 0:
            features.append(np.mean(((axis - mean) / std) ** 3))
            features.append(np.mean(((axis - mean) / std) ** 4))
        else:
            features.append(0.0)
            features.append(0.0)
    
    # SMA
    sma = (np.sum(np.abs(x)) + np.sum(np.abs(y)) + np.sum(np.abs(z))) / len(x)
    features.append(sma)
    
    # Magnitude features
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    features.append(np.mean(magnitude))
    features.append(np.max(magnitude))
    features.append(np.std(magnitude))
    
    return np.array(features)

# ─────────────────────────────────────────────
# APP UI
# ─────────────────────────────────────────────
st.title("🏥 Edge AI Fall Detection System")
st.markdown("**Elderly Healthcare Monitoring** — Powered by Quantised TFLite Neural Network")
st.caption(f"Classification threshold: **{THRESHOLD}** — optimised for high Fall Recall")

st.sidebar.success("✅ Model loaded successfully")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model Performance**\n\n"
    "- Fall Recall: 0.836\n"
    "- Fall Precision: 0.815\n"
    "- F1-Score: 0.825\n"
    "- Accuracy: 88.46%\n"
    "- Latency: ~0.011ms"
)

# Input Method
input_method = st.radio(
    "Choose input method:",
    ["Manual Feature Input", "Upload Raw Sensor CSV"],
    horizontal=True
)

st.markdown("---")

# === MANUAL INPUT ===
if input_method == "Manual Feature Input":
    st.subheader("Enter 28 Extracted Feature Values")
    cols = st.columns(4)
    user_input = {}
    
    for i, feature in enumerate(FEATURE_NAMES):
        with cols[i % 4]:
            user_input[feature] = st.number_input(
                label=feature,
                value=0.0,
                format="%.4f",
                key=feature
            )
    
    if st.button("🔍 Predict Fall Event", type="primary", use_container_width=True):
        features = np.array([user_input[f] for f in FEATURE_NAMES])
        probability, prediction = predict_tflite(interpreter, scaler, features)
        
        # Display Result
        st.subheader("Prediction Result")
        col1, col2, col3 = st.columns(3)
        with col1:
            if prediction == 1:
                st.error("⚠️ FALL DETECTED")
            else:
                st.success("✅ NO FALL DETECTED")
        with col2:
            st.metric("Fall Probability", f"{probability*100:.2f}%")
        with col3:
            st.metric("Threshold", f"{THRESHOLD}")
        
        st.progress(float(probability))

# === CSV UPLOAD ===
else:
    st.subheader("Upload Raw Accelerometer CSV")
    uploaded_csv = st.file_uploader("Upload CSV file (columns: X, Y, Z)", type=["csv"])
    
    if uploaded_csv is not None:
        try:
            raw_df = pd.read_csv(uploaded_csv)
            if not all(col in raw_df.columns for col in ['X', 'Y', 'Z']):
                st.error("CSV must contain columns: X, Y, Z")
            else:
                st.success(f"File loaded — {len(raw_df)} rows")
                st.line_chart(raw_df[['X', 'Y', 'Z']].head(300))
                
                if st.button("Extract Features & Predict", type="primary", use_container_width=True):
                    with st.spinner("Extracting features..."):
                        features = extract_features_from_raw(raw_df)
                    
                    probability, prediction = predict_tflite(interpreter, scaler, features)
                    
                    # Show extracted features
                    feat_df = pd.DataFrame([features], columns=FEATURE_NAMES).round(4)
                    st.dataframe(feat_df, use_container_width=True)
                    
                    # Result
                    st.subheader("Prediction Result")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if prediction == 1:
                            st.error("⚠️ FALL DETECTED")
                        else:
                            st.success("✅ NO FALL DETECTED")
                    with col2:
                        st.metric("Fall Probability", f"{probability*100:.2f}%")
                    with col3:
                        st.metric("Threshold", THRESHOLD)
                    
                    st.progress(float(probability))
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Footer
st.markdown("---")
st.caption("⚠️ For research/educational purposes only. Not a medical device.")
st.caption("SisFall Dataset • Quantised TFLite Model")