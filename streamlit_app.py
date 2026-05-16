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
import os

# ─────────────────────────────────────────────
# TFLITE RUNTIME IMPORT
# Uses lightweight tflite-runtime instead of
# full TensorFlow to reduce deployment size
# ─────────────────────────────────────────────
"""
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from tensorflow import lite as tflite
    except ImportError:
        st.error(
            "Could not import tflite_runtime or tensorflow. "
            "Please check requirements.txt"
        )
        st.stop()
"""
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
# FEATURE NAMES
# ─────────────────────────────────────────────

FEATURE_NAMES = []
for axis in ['X', 'Y', 'Z']:
    for stat in ['Mean', 'Std', 'Min', 'Max',
                 'Range', 'RMS', 'Skewness', 'Kurtosis']:
        FEATURE_NAMES.append(f"{axis}_{stat}")
FEATURE_NAMES += ['SMA', 'Magnitude_Mean',
                  'Peak_Acceleration', 'Magnitude_Std']

THRESHOLD = 0.3

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
""""
@st.cache_resource
def load_model():
    try:
        interpreter = tflite.Interpreter(
            model_path="model_quantised.tflite")
        interpreter.allocate_tensors()
        scaler = joblib.load("scaler.pkl")
        return interpreter, scaler, None
    except Exception as e:
        return None, None, str(e)
"""
interpreter = Interpreter(model_path="model_quantised.tflite")
interpreter.allocate_tensors()
# ─────────────────────────────────────────────
# INFERENCE FUNCTION
# ─────────────────────────────────────────────

def predict_tflite(interpreter, scaler, features):
    features_scaled = scaler.transform(
        features.reshape(1, -1)).astype(np.float32)
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(
        input_details[0]['index'], features_scaled)
    interpreter.invoke()
    probability = interpreter.get_tensor(
        output_details[0]['index'])[0][0]
    prediction = 1 if probability >= THRESHOLD else 0
    return float(probability), prediction

# ─────────────────────────────────────────────
# FEATURE EXTRACTION FROM RAW CSV
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
        std  = np.std(axis)
        if std > 0:
            features.append(
                np.mean(((axis - mean) / std) ** 3))
            features.append(
                np.mean(((axis - mean) / std) ** 4))
        else:
            features.append(0.0)
            features.append(0.0)
    sma = (np.sum(np.abs(x)) +
           np.sum(np.abs(y)) +
           np.sum(np.abs(z))) / len(x)
    features.append(sma)
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    features.append(np.mean(magnitude))
    features.append(np.max(magnitude))
    features.append(np.std(magnitude))
    return np.array(features)

# ─────────────────────────────────────────────
# APP HEADER
# ─────────────────────────────────────────────

st.title("🏥 Edge AI Fall Detection System")
st.markdown(
    "**Elderly Healthcare Monitoring** — "
    "Powered by Quantised TFLite Neural Network"
)
st.markdown(
    f"*Classification threshold: {THRESHOLD} "
    f"— optimised for Fall Recall = 0.8213*"
)
st.markdown("---")

# ─────────────────────────────────────────────
# LOAD MODEL AT STARTUP
# ─────────────────────────────────────────────

interpreter, scaler, error = load_model()

if interpreter is None:
    st.error(
        f"Failed to load model: {error}. "
        "Please ensure model_quantised.tflite and "
        "scaler.pkl are in the repository root."
    )
    st.stop()
else:
    st.sidebar.success("✅ Model loaded successfully")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model Performance:**\n\n"
    "- Fall Recall: 0.836\n"
    "- Fall Precision: 0.815\n"
    "- F1-Score: 0.825\n"
    "- Accuracy: 88.46%\n"
    "- Latency: 0.011ms\n"
    "- Threshold: 0.3"
)

# ─────────────────────────────────────────────
# INPUT METHOD SELECTION
# ─────────────────────────────────────────────

st.subheader("Input Method")
input_method = st.radio(
    "Choose how to provide sensor data:",
    ["Manual Feature Input", "Upload Raw Sensor CSV"],
    horizontal=True
)
st.markdown("---")

# ─────────────────────────────────────────────
# METHOD 1: MANUAL FEATURE INPUT
# ─────────────────────────────────────────────

if input_method == "Manual Feature Input":
    st.subheader("Enter 28 Extracted Feature Values")
    st.markdown(
        "*Enter the pre-extracted feature values "
        "from a 3-second accelerometer window.*"
    )
    default_values = {
        'X_Mean': 0.021, 'X_Std': 0.187,
        'X_Min': -2.31, 'X_Max': 2.45,
        'X_Range': 4.76, 'X_RMS': 0.189,
        'X_Skewness': 0.12, 'X_Kurtosis': 3.21,
        'Y_Mean': 0.015, 'Y_Std': 0.143,
        'Y_Min': -1.87, 'Y_Max': 1.92,
        'Y_Range': 3.79, 'Y_RMS': 0.144,
        'Y_Skewness': -0.08, 'Y_Kurtosis': 2.98,
        'Z_Mean': 0.981, 'Z_Std': 0.201,
        'Z_Min': -1.23, 'Z_Max': 2.87,
        'Z_Range': 4.10, 'Z_RMS': 1.003,
        'Z_Skewness': 0.31, 'Z_Kurtosis': 3.45,
        'SMA': 1.24, 'Magnitude_Mean': 1.31,
        'Peak_Acceleration': 4.87,
        'Magnitude_Std': 0.342
    }
    user_input = {}
    cols = st.columns(4)
    for i, feature in enumerate(FEATURE_NAMES):
        with cols[i % 4]:
            user_input[feature] = st.number_input(
                label=feature,
                value=float(
                    default_values.get(feature, 0.0)),
                format="%.4f",
                key=feature
            )
    st.markdown("---")
    if st.button("Predict Fall Event",
                 type="primary",
                 use_container_width=True):
        features = np.array(
            [user_input[f] for f in FEATURE_NAMES])
        probability, prediction = predict_tflite(
            interpreter, scaler, features)
        st.markdown("---")
        st.subheader("Prediction Result")
        col1, col2, col3 = st.columns(3)
        with col1:
            if prediction == 1:
                st.error("⚠️ FALL DETECTED")
            else:
                st.success("✅ NO FALL DETECTED")
        with col2:
            st.metric("Fall Probability",
                      f"{probability*100:.2f}%")
        with col3:
            st.metric("Threshold Used", f"{THRESHOLD}")
        st.progress(float(probability))
        st.caption(
            f"Probability: {probability:.4f} — "
            f"{'Above' if probability >= THRESHOLD else 'Below'}"
            f" threshold of {THRESHOLD}"
        )

# ─────────────────────────────────────────────
# METHOD 2: CSV UPLOAD
# ─────────────────────────────────────────────

else:
    st.subheader("Upload Raw Accelerometer CSV")
    st.markdown(
        "*Upload a CSV file with columns **X**, **Y**, **Z** "
        "in gravitational units (g). "
        "Minimum 600 rows recommended.*"
    )
    st.markdown("**Expected CSV format:**")
    sample_df = pd.DataFrame({
        'X': [0.021, 0.019, 0.023],
        'Y': [0.015, 0.017, 0.014],
        'Z': [0.981, 0.979, 0.983]
    })
    st.dataframe(sample_df, use_container_width=False)
    uploaded_csv = st.file_uploader(
        "Upload sensor CSV file", type=["csv"])
    if uploaded_csv is not None:
        try:
            raw_df = pd.read_csv(uploaded_csv)
            required_cols = ['X', 'Y', 'Z']
            if not all(c in raw_df.columns
                       for c in required_cols):
                st.error(
                    "CSV must contain columns: X, Y, Z. "
                    f"Found: {list(raw_df.columns)}"
                )
            else:
                st.success(
                    f"File loaded — {len(raw_df)} rows")
                st.dataframe(
                    raw_df.head(10),
                    use_container_width=True)
                st.markdown("**Signal Preview:**")
                st.line_chart(
                    raw_df[['X', 'Y', 'Z']].head(200))
                if st.button(
                        "Extract Features & Predict",
                        type="primary",
                        use_container_width=True):
                    with st.spinner(
                            "Extracting features..."):
                        features = \
                            extract_features_from_raw(
                                raw_df)
                    st.markdown(
                        "**Extracted Features:**")
                    feat_df = pd.DataFrame(
                        [features],
                        columns=FEATURE_NAMES
                    ).round(4)
                    st.dataframe(
                        feat_df,
                        use_container_width=True)
                    probability, prediction = \
                        predict_tflite(
                            interpreter,
                            scaler,
                            features)
                    st.markdown("---")
                    st.subheader("Prediction Result")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if prediction == 1:
                            st.error("⚠️ FALL DETECTED")
                        else:
                            st.success(
                                "✅ NO FALL DETECTED")
                    with col2:
                        st.metric(
                            "Fall Probability",
                            f"{probability*100:.2f}%")
                    with col3:
                        st.metric(
                            "Threshold Used",
                            f"{THRESHOLD}")
                    st.progress(float(probability))
                    st.caption(
                        f"Probability: {probability:.4f}"
                        f" — "
                        f"{'Above' if probability >= THRESHOLD else 'Below'}"
                        f" threshold of {THRESHOLD}"
                    )
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")
st.caption(
    "⚠️ This system is intended for research and "
    "educational purposes only. It is not a substitute "
    "for professional medical diagnosis or emergency "
    "services. In a real emergency always call 999."
)
st.caption(
    "Edge AI Fall Detection System — "
    "SisFall Dataset (Sucerquia et al., 2017) — "
    "TFLite Quantised Neural Network"
)