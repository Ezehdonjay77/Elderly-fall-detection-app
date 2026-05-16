import streamlit as st
import numpy as np
import os
import joblib
import tensorflow as tf

# Load scaler
scaler = joblib.load("scaler.pkl")

# Load TFLite model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_quantised.tflite")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("🧓 Fall Detection System (Edge AI)")

st.write("Enter sensor feature values (from accelerometer model)")

# Example input (replace with real sensor pipeline later)
features = st.text_input("Enter 20 feature values (comma-separated)")

if st.button("Predict Fall"):
    try:
        data = np.array([float(x) for x in features.split(",")]).reshape(1, -1)

        # scale
        data = scaler.transform(data)

        # inference
        interpreter.set_tensor(input_details[0]['index'], data.astype(np.float32))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        pred = output[0][0]

        if pred > 0.5:
            st.error("⚠️ FALL DETECTED")
        else:
            st.success("Normal activity")

    except Exception as e:
        st.error(f"Error: {e}")