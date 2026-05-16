import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model_quantised.tflite")
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

interpreter = load_model()
scaler = load_scaler()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ─────────────────────────────────────────────
# FEATURE ENGINEERING (IDENTICAL TO TRAINING)
# ─────────────────────────────────────────────

def extract_features(window):
    x = window[:, 0]
    y = window[:, 1]
    z = window[:, 2]

    features = []

    for axis in [x, y, z]:
        mean = np.mean(axis)
        std = np.std(axis)

        features.append(mean)
        features.append(std)
        features.append(np.min(axis))
        features.append(np.max(axis))
        features.append(np.max(axis) - np.min(axis))
        features.append(np.sqrt(np.mean(axis ** 2)))

        if std > 0:
            z_axis = (axis - mean) / std
            features.append(np.mean(z_axis ** 3))  # skewness
            features.append(np.mean(z_axis ** 4))  # kurtosis
        else:
            features.append(0.0)
            features.append(0.0)

    magnitude = np.sqrt(x**2 + y**2 + z**2)

    sma = (np.sum(np.abs(x)) + np.sum(np.abs(y)) + np.sum(np.abs(z))) / len(x)

    features.append(sma)
    features.append(np.mean(magnitude))
    features.append(np.max(magnitude))
    features.append(np.std(magnitude))

    return np.array(features)

# ─────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────

def predict_fall(data):
    features = extract_features(data).reshape(1, -1)

    # scale (VERY IMPORTANT)
    features = scaler.transform(features)

    # TFLite inference
    interpreter.set_tensor(input_details[0]["index"], features.astype(np.float32))
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0][0]

    return output

# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

st.set_page_config(page_title="Fall Detection AI", layout="centered")

st.title("🚨 Elderly Fall Detection System (Edge AI)")
st.write("Upload a CSV file containing accelerometer data (X, Y, Z columns)")

uploaded_file = st.file_uploader("Upload sensor data", type=["csv"])

if uploaded_file is not None:

    try:
        data = np.loadtxt(uploaded_file, delimiter=",")

        if data.shape[1] != 3:
            st.error("CSV must contain exactly 3 columns: X, Y, Z")
            st.stop()

        st.success(f"Data loaded: {data.shape[0]} samples")

        if len(data) < 50:
            st.warning("Very small dataset — prediction may be unreliable")

        # Predict
        score = predict_fall(data)

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Fall Probability", f"{score:.2f}")

        with col2:
            if score > 0.5:
                st.error("⚠️ FALL DETECTED")
            else:
                st.success("✅ No Fall Detected")

        st.progress(float(score))

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")