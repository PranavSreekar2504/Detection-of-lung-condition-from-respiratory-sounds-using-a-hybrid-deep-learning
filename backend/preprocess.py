import librosa
import numpy as np
from PIL import Image
import io
import noisereduce as nr

# Configuration
SAMPLE_RATE = 22050
N_MELS = 128
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
FMAX = 4000

def load_audio(audio_bytes):
    """
    Load and resample audio from bytes to uniform sampling rate
    
    Args:
        audio_bytes: Audio file in bytes
    
    Returns:
        y: Time series audio data
        sr: Sampling rate
    """
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
    return y, sr

def remove_silence(y, sr, threshold_db=40):
    """
    Remove silence from audio using librosa's trim function
    
    Args:
        y: Audio time series
        sr: Sampling rate
        threshold_db: Threshold for silence removal
    
    Returns:
        y_trimmed: Audio with silence removed
    """
    y_trimmed, _ = librosa.effects.trim(y, top_db=threshold_db)
    return y_trimmed

def normalize_amplitude(y):
    """
    Normalize audio amplitude to [-1, 1] range
    
    Args:
        y: Audio time series
    
    Returns:
        y_normalized: Normalized audio
    """
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y_normalized = y / max_val
    else:
        y_normalized = y
    return y_normalized

def reduce_noise(y, sr):
    """
    Apply noise reduction using spectral subtraction
    
    Args:
        y: Audio time series
        sr: Sampling rate
    
    Returns:
        y_reduced: Denoised audio
    """
    # Reduce noise using noisereduce library
    y_reduced = nr.reduce_noise(y=y, sr=sr)
    return y_reduced

def extract_mel_spectrogram(y, sr):
    """
    Extract Log-Mel Spectrogram as 2D representation
    
    Args:
        y: Audio time series
        sr: Sampling rate
    
    Returns:
        mel_spec_db: Log-Mel spectrogram
    """
    # Compute mel-scaled spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmax=FMAX
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

def extract_mfcc(y, sr):
    """
    Extract MFCC (Mel-Frequency Cepstral Coefficients) features
    
    Args:
        y: Audio time series
        sr: Sampling rate
    
    Returns:
        mfcc: MFCC features
    """
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    return mfcc

def spectrogram_to_image(mel_spec_db):
    """
    Convert mel-spectrogram to 3-channel RGB image for CNN input
    
    Args:
        mel_spec_db: Log-Mel spectrogram
    
    Returns:
        img: PIL Image of size (224, 224, 3)
    """
    # Normalize to 0-255 range
    mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
    mel_spec_scaled = (mel_spec_normalized * 255).astype(np.uint8)
    
    # Convert to PIL Image
    img = Image.fromarray(mel_spec_scaled).convert("RGB")
    
    # Resize to standard CNN input size (224x224)
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    
    return img

def preprocess_audio(audio_bytes):
    """
    Complete preprocessing pipeline
    
    Steps:
    1. Load audio and resample to 22050 Hz
    2. Remove silence
    3. Apply noise reduction
    4. Normalize amplitude
    5. Extract mel-spectrogram
    6. Convert to RGB image
    
    Args:
        audio_bytes: Audio file in bytes
    
    Returns:
        img: Processed image tensor ready for model
        mel_spec_db: Mel-spectrogram for visualization
        mfcc: MFCC features
    """
    # Step 1: Load and resample
    y, sr = load_audio(audio_bytes)
    
    # Step 2: Remove silence
    y = remove_silence(y, sr)
    
    # Step 3: Noise reduction
    y = reduce_noise(y, sr)
    
    # Step 4: Normalize
    y = normalize_amplitude(y)
    
    # Step 5: Extract features
    mel_spec_db = extract_mel_spectrogram(y, sr)
    mfcc = extract_mfcc(y, sr)
    
    # Step 6: Convert to image
    img = spectrogram_to_image(mel_spec_db)
    
    return img, mel_spec_db, mfcc, y, sr