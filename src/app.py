import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from src.model import HybridResNetLungDetector
from src.preprocess import preprocess_audio
import torch
from PIL import Image
import io

# Load model
@st.cache_resource
def load_model():
    model = HybridResNetLungDetector(num_classes=6)
    model.load_state_dict(torch.load('models/best_model.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

st.title("Lung Disease Detection from Respiratory Sounds")
st.write("Upload an audio file to detect lung conditions using our hybrid deep learning model.")

uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'flac'])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Preprocess audio
    segments, mel_spec, mfcc, audio, sr = preprocess_audio("temp_audio.wav")
    
    # Display audio waveform
    st.subheader("Audio Waveform")
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)
    
    # Display spectrogram
    st.subheader("Mel Spectrogram")
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mel_spec, x_axis='time', y_axis='mel', sr=sr, ax=ax)
    ax.set_title("Mel Spectrogram")
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    st.pyplot(fig)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(segments)
        probabilities = torch.softmax(outputs, dim=1)
        avg_probabilities = probabilities.mean(dim=0)
        predicted_class = torch.argmax(avg_probabilities).item()
    
    # Display results
    st.subheader("Prediction Results")
    class_names = model.CLASS_NAMES
    st.write(f"**Predicted Condition:** {class_names[predicted_class]}")
    st.write(f"**Description:** {model.CLASS_DESCRIPTIONS[class_names[predicted_class]]}")
    st.write(f"**Recommendation:** {model.CLASS_RECOMMENDATIONS[class_names[predicted_class]]}")
    
    # Confidence scores
    st.subheader("Confidence Scores")
    confidence_scores = avg_probabilities.numpy()
    for i, (class_name, score) in enumerate(zip(class_names, confidence_scores)):
        st.write(f"{class_name}: {score:.4f}")
    
    # Bar chart of probabilities
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(class_names, confidence_scores)
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probabilities')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Clean up
    import os
    os.remove("temp_audio.wav")