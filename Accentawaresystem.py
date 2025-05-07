import streamlit as st
import numpy as np
import joblib
import soundfile as sf
import librosa
import tempfile
import os
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(layout="wide")
st.title("🎙️ Accent Detection System - Simplified Version")

# Sidebar Debug Toggle
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=True)

# Load Model
try:
    model = joblib.load("accent_model.pkl")
    st.sidebar.success("✅ Model loaded successfully")
    if debug_mode:
        st.sidebar.subheader("Model Info")
        st.sidebar.write(f"Model Type: {type(model).__name__}")
        if hasattr(model, 'classes_'):
            st.sidebar.write(f"Classes: {model.classes_}")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load model: {e}")
    model = None

# Load Label Encoder
try:
    label_encoder = joblib.load("label_encoder.pkl")
    st.sidebar.success("✅ Label Encoder loaded successfully")
    if debug_mode:
        st.sidebar.subheader("Encoder Info")
        st.sidebar.write(f"Classes: {label_encoder.classes_}")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load encoder: {e}")
    label_encoder = None

# Stop if critical assets fail
if model is None or label_encoder is None:
    st.error("Cannot proceed without model and encoder.")
    st.stop()

# Feature extraction
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        features = np.hstack([np.mean(mfccs, axis=1), spectral_centroid, spectral_rolloff, zcr])
        
        # Ensure 12 features
        if features.shape[0] > 12:
            features = features[:12]
        elif features.shape[0] < 12:
            features = np.pad(features, (0, 12 - features.shape[0]), 'constant')
        
        return features.reshape(1, -1)
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None

# Upload and prediction section
st.subheader("🎧 Upload Audio File")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name

    # Feature extraction
    features = extract_features(temp_path)

    if features is not None:
        try:
            pred = model.predict(features)
            proba = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None
            predicted_label = label_encoder.inverse_transform(pred)[0]

            st.success(f"🗣️ Predicted Accent: **{predicted_label}**")

            if proba is not None:
                proba_df = pd.DataFrame({
                    "Accent": label_encoder.classes_,
                    "Probability (%)": proba * 100
                }).sort_values("Probability (%)", ascending=False)

                # Single Unified Visualization
                st.subheader("📊 Prediction Probability Distribution")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(proba_df["Accent"], proba_df["Probability (%)"], color='skyblue')
                ax.set_xlabel("Probability (%)")
                ax.set_title("Accent Prediction Confidence")
                ax.invert_yaxis()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction error: {e}")

    os.unlink(temp_path)

# Custom Test Section
st.subheader("🧪 Test with Custom Feature Vector (Optional)")
example_vector = np.array([[7.07, -6.51, 7.65, 11.15, -7.65, 12.48,
                            -11.70, 3.42, 1.46, -2.81, 0.86, -5.24]])
default_str = ", ".join(map(str, example_vector[0]))

custom_input_str = st.text_area("Enter comma-separated feature values (12 values):", value=default_str)

try:
    custom_features = np.array([float(x.strip()) for x in custom_input_str.split(",")]).reshape(1, -1)
    if custom_features.shape[1] == 12 and st.button("🔍 Predict from Custom Input"):
        pred = model.predict(custom_features)
        proba = model.predict_proba(custom_features)[0] if hasattr(model, 'predict_proba') else None
        predicted_label = label_encoder.inverse_transform(pred)[0]
        
        st.success(f"🗣️ Predicted Accent: **{predicted_label}**")
        
        if proba is not None:
            proba_df = pd.DataFrame({
                "Accent": label_encoder.classes_,
                "Probability (%)": proba * 100
            }).sort_values("Probability (%)", ascending=False)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(proba_df["Accent"], proba_df["Probability (%)"], color='lightgreen')
            ax.set_xlabel("Probability (%)")
            ax.set_title("Custom Feature Input: Prediction Confidence")
            ax.invert_yaxis()
            st.pyplot(fig)
except Exception as e:
    st.error(f"Custom input error: {e}")
