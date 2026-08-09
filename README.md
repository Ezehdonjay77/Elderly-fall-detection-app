# 🏥 Edge AI Elderly Fall Detection System

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-TFLite-orange.svg)](https://www.tensorflow.org/lite)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen.svg)](https://elderly-fall-detection-app-duemrb5vpwfmvdkgydlmgf.streamlit.app/)
[![Edge AI](https://img.shields.io/badge/Edge%20AI-TFLite%20Quantised-purple.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A full Edge AI pipeline for real-time elderly fall detection using wearable accelerometer data. This project covers supervised ML classification, lightweight neural network design, TensorFlow Lite conversion, post-training quantisation, and magnitude-based weight pruning — optimised for deployment on memory-constrained wearable devices.

🔗 **[Live Demo — Try the App](https://elderly-fall-detection-app-duemrb5vpwfmvdkgydlmgf.streamlit.app/)**  
📓 **[Google Colab Notebook](https://colab.research.google.com/drive/1fJfqPZJQkiYL9Gy8Nsqp9I0lPLf1aUbh?usp=sharing)**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [Methodology](#methodology)
- [ML Classification Results](#ml-classification-results)
- [Edge Optimisation](#edge-optimisation)
- [Latency and Efficiency](#latency-and-efficiency)
- [Web Deployment](#web-deployment)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Ethical Considerations](#ethical-considerations)
- [Future Work](#future-work)

---

## Overview

Falls are the leading cause of injury-related deaths among adults aged 65 and above, with approximately 684,000 fatal falls occurring globally each year (WHO, 2023). Existing fall detection systems rely on cloud-based infrastructure, introducing three critical limitations: network-dependent latency that delays emergency alerts, high bandwidth and battery consumption, and serious data privacy risks under UK GDPR when transmitting sensitive biometric health data.

This project addresses these limitations by designing and implementing a complete **Edge AI fall detection pipeline** that runs entirely on a wearable device. All inference is performed locally — only a binary fall alert is transmitted via Bluetooth Low Energy (BLE) upon detection, ensuring maximum data minimisation and UK GDPR compliance.

**Research Hypotheses:**
- *H0:* Wearable accelerometer data processed using ML models can accurately detect fall events
- *H1:* Edge AI architecture eliminates cloud-based latency, enabling sub-10ms real-time inference on wearable hardware

Both hypotheses were confirmed.

---

## Dataset

**Source:** [SisFall Dataset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5298771/) — Sucerquia et al. (2017)

| Property | Value |
|---|---|
| Volunteers | 38 (23 younger adults aged 19–30, 15 elderly aged 60–75) |
| Activity types | 34 (19 Activities of Daily Living + 15 fall types) |
| Sensor | ADXL345 tri-axial accelerometer (±16g, 13-bit resolution) |
| Sampling rate | 200Hz |
| Sensor placement | Waist-worn |
| Total windows (after segmentation) | 44,093 (29,707 Non-Fall / 14,386 Fall) |
| Class labels | 1 = Fall, 0 = Non-Fall (ADL) |

Only the ADXL345 accelerometer was used as the primary sensor, consistent with low-power edge deployment requirements.

---

## System Architecture

The pipeline follows five stages, all designed for local execution on the target wearable device:

```
Wearable Sensor (ADXL345)
        ↓
Data Acquisition (X, Y, Z axes — local storage)
        ↓
Preprocessing (Sliding window: 3s, 50% overlap, bit conversion)
        ↓
Feature Extraction (28 features: 24 per-axis + 4 combined)
        ↓
ML Inference (TFLite Quantised — threshold = 0.3 — 0.036ms latency)
        ↓
Edge Optimisation Layer
    ├── Quantisation (float32 → int8, ~4x size reduction)
    └── Pruning (30% sparsity, threshold = 0.4)
        ↓
Classification Decision
    ├── Fall Detected → BLE Alert Transmitted
    └── No Fall → Continue Monitoring
```

All processing is local. Raw sensor data never leaves the device — UK GDPR compliant.

---

## Methodology

### Feature Extraction

Raw accelerometer readings were segmented using a sliding window approach:
- **Window size:** 3 seconds (600 samples at 200Hz)
- **Overlap:** 50% (step size of 300 samples)
- **Conversion:** Raw bit values converted to gravitational units using the ADXL345 manufacturer formula

**28 features extracted per window:**

| Feature Group | Count | Features |
|---|---|---|
| Per-axis statistical (X, Y, Z) | 24 | Mean, Std Dev, Min, Max, Range, RMS, Skewness, Kurtosis |
| Combined signal features | 4 | SMA, Acceleration Magnitude Mean, Peak Acceleration, Magnitude Std Dev |

These features capture both the free-fall phase (reduced acceleration magnitude) and the impact phase (sharp acceleration increase) — together forming the distinctive signature of a fall event.

### Model Training

- Stratified 80/20 train-test split preserving fall/non-fall ratio
- All classifiers implemented within scikit-learn Pipelines with StandardScaler to prevent data leakage
- Stratified 5-Fold Cross-Validation on the training set

### Models Implemented

| Model | Rationale |
|---|---|
| Logistic Regression | Interpretable baseline |
| SVM (RBF Kernel) | Strong performance on high-dimensional biomedical data |
| Random Forest | Best generalisation; robust ensemble method |

---

## ML Classification Results

### Cross-Validation Performance (Training Set)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8384 | 0.8545 | 0.6081 | 0.7105 | 0.8287 |
| SVM (RBF) | 0.8598 | 0.9236 | 0.6217 | 0.7431 | 0.9117 |
| **Random Forest** | **0.9121** | **0.9432** | **0.7773** | **0.8522** | **0.9688** |

### Held-Out Test Set Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8436 | 0.8615 | 0.6204 | 0.7214 | 0.8284 |
| SVM (RBF) | 0.8646 | 0.9278 | 0.6343 | 0.7535 | 0.9204 |
| **Random Forest** | **0.9221** | **0.9451** | **0.8081** | **0.8713** | **0.9717** |

Random Forest achieved the strongest performance across all metrics, confirming hypothesis H0. Despite this, its serialised model size (36,285 KB) made it unsuitable for direct wearable deployment, motivating the lightweight neural network approach.

---

## Edge Optimisation

### Lightweight Neural Network Architecture

A compact neural network was designed specifically for edge deployment:

| Layer | Configuration |
|---|---|
| Input | 28 features (30 after preprocessing) |
| Hidden Layer 1 | Dense(64) → BatchNorm → ReLU → Dropout(0.5) |
| Hidden Layer 2 | Dense(32) → BatchNorm → ReLU → Dropout(0.5) |
| Output | Dense(1) → Sigmoid |
| Optimiser | Adam |
| Loss | Binary Cross-Entropy |
| Training | 30 epochs, batch size 32 |

### Threshold Optimisation

The default threshold of 0.5 yielded a clinically insufficient Fall Recall of 0.708. Threshold analysis identified **0.3 as the optimal configuration**:

| Threshold | Fall Recall | Fall Precision | F1-Score | Accuracy |
|---|---|---|---|---|
| 0.5 | 0.7084 | 0.9426 | 0.8089 | 0.8908 |
| 0.4 | 0.7483 | 0.9181 | 0.8246 | 0.8961 |
| **0.3** | **0.8213** | **0.8131** | **0.8172** | **0.8801** |
| 0.2 | 0.9385 | 0.6203 | 0.7469 | 0.7925 |

Threshold 0.3 was selected — it balances clinical sensitivity with acceptable false alarm rates. Threshold 0.2 achieved higher recall but caused precision to collapse to 0.62, producing clinically unacceptable alert fatigue.

### Model Size Comparison

| Model | Size (KB) | Size Reduction | Edge Suitable |
|---|---|---|---|
| Random Forest | 36,285 | Baseline | ❌ |
| Full Neural Network (float32) | 83.3 | — | ⚠️ |
| TFLite float32 | 17.7 | 78.8% | ✅ |
| **TFLite Quantised (int8)** | **7.6** | **90.9%** | **✅** |
| Pruned + Quantised TFLite | 7.6 | 90.9% | ✅ |

### Weight Pruning Results

Magnitude-based pruning experiments across 30%–80% sparsity showed that **30% sparsity at threshold 0.4** is the optimal pruned configuration:

- Fall Recall: 0.75
- Precision: 0.88
- F1-Score: 0.81

Sparsity above 50% degraded Fall Recall to clinically unacceptable levels.

---

## Latency and Efficiency

### Inference Latency (100-run average)

| Model | Avg Latency (ms) | Reduction |
|---|---|---|
| Full Neural Network | 91.198 | Baseline |
| **Quantised TFLite** | **0.036** | **~100%** |

The quantised model satisfies the sub-10ms edge deployment target. Measurements were conducted in Google Colab — real wearable deployment is expected to yield 1–5ms, still well within the real-time detection window.

### Data Transmission Reduction

| System | Data Transmission |
|---|---|
| Cloud-based (raw streaming) | ~8,437 KB/hr |
| Edge AI (BLE binary alert only) | Near zero |

Transmitting only a binary fall alert via BLE reduces data transmission by several orders of magnitude, directly reducing bandwidth consumption, cloud energy cost, and wearable battery drain.

---

## Web Deployment

A Streamlit application was developed to demonstrate the fall detection pipeline:

- Accepts preprocessed accelerometer feature inputs
- Returns fall / no-fall classification with confidence score
- Powered by the trained model

🔗 **[Launch the App](https://elderly-fall-detection-app-duemrb5vpwfmvdkgydlmgf.streamlit.app/)**

---

## Project Structure

```
elderly-fall-detection-app/
│
├── app.py                     # Streamlit web application
├── requirements.txt           # Dependencies
├── model/
│   ├── random_forest.pkl      # Trained Random Forest model
│   ├── fall_detection_nn.h5   # Full neural network
│   └── fall_detection.tflite  # Quantised TFLite model
├── notebooks/
│   └── fall_detection.ipynb   # Full experimental notebook
├── data/
│   └── sisfall_processed.csv  # Processed feature dataset
├── figures/
│   ├── class_distribution.png
│   ├── acceleration_magnitude.png
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   ├── threshold_analysis.png
│   └── edge_pipeline.png
└── README.md
```

---

## How to Run

### Run Locally

```bash
# Clone the repository
git clone https://github.com/Ezehdonjay77/Elderly-fall-detection-app.git
cd Elderly-fall-detection-app

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py
```

### Run the Notebook

Open directly in Google Colab:  
[📓 Open in Colab](https://colab.research.google.com/drive/1fJfqPZJQkiYL9Gy8Nsqp9I0lPLf1aUbh?usp=sharing)

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python | Core language |
| scikit-learn | ML models, pipelines, cross-validation, metrics |
| TensorFlow / Keras | Neural network training and TFLite conversion |
| tensorflow-model-optimization | Magnitude-based weight pruning |
| numpy | Numerical computations and feature calculations |
| pandas | Data loading and manipulation |
| matplotlib / seaborn | Visualisation and confusion matrices |
| joblib | Saving and loading trained sklearn models |
| Streamlit | Web application and deployment |
| Google Colab | Development environment |

---

## Ethical Considerations

- **Data Privacy:** The SisFall dataset is publicly available and fully anonymised. All processing in the proposed system runs locally on the wearable — raw data never leaves the device
- **UK GDPR Compliance:** Biometric health data is classified as special category data under Article 9. The Edge AI framework satisfies data minimisation and purpose limitation principles by transmitting only binary alerts via BLE
- **Algorithmic Bias:** The SisFall dataset has limited ethnic and gender representation. Models may exhibit reduced accuracy on underrepresented demographics — future work should include more diverse training data
- **Explainability:** Feature importance analysis provides clinicians and caregivers with interpretable evidence supporting model decisions, aligning with explainable AI principles in high-stakes healthcare settings
- **Clinical Role:** The system is intended as a detection and alerting tool only — it does not replace clinical assessment or medical diagnosis

---

## Future Work

- **Quantisation-Aware Training (QAT)** — applying quantisation constraints during training to recover 3–5% accuracy lost during post-training conversion
- **On-device validation** on physical hardware (Raspberry Pi Zero or Arduino Nano 33 BLE Sense) for realistic latency and energy measurements
- **Knowledge distillation** — training a compact student model to replicate Random Forest's decision boundaries at neural network size
- **Multi-sensor fusion** — combining gyroscope data with accelerometer readings to better differentiate near-fall scenarios from actual falls
- **Dataset diversity expansion** to reduce demographic bias and improve generalisation across broader elderly populations
- **Federated learning** — enabling privacy-preserving model improvements across multiple wearable devices without transmitting raw sensor data

---

## References

- Sucerquia, A., López, J. D., & Vargas-Bonilla, J. F. (2017). SisFall: A fall and movement dataset. *Sensors*, 17(1), 198.
- Özdemir, A. T., & Barshan, B. (2014). Detecting falls with wearable sensors using machine learning techniques. *Sensors*, 14(6), 10691–10708.
- Nguyen, D. C. et al. (2023). Edge AI: On-demand accelerating deep neural network inference via edge computing. *IEEE Communications Surveys & Tutorials*, 25(1), 123–163.
- Shi, W. et al. (2016). Edge computing: Vision and challenges. *IEEE Internet of Things Journal*, 3(5), 637–646.
- Florence, C. S. et al. (2018). Medical costs of fatal and nonfatal falls in older adults. *American Journal of Preventive Medicine*, 54(2), 169–177.

---

*Developed by John Ifeanyichukwu Ezeh — MSc Data Science and Artificial Intelligence*
