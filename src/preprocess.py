import librosa
import numpy as np
import torch
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')

def preprocess_audio(audio_path, target_sr=22050, segment_length=3.0, overlap=0.5):
    """
    Preprocess audio file for model input.
    
    Args:
        audio_path (str): Path to audio file
        target_sr (int): Target sample rate
        segment_length (float): Length of each segment in seconds
        overlap (float): Overlap between segments as fraction
    
    Returns:
        tuple: (segments, mel_spectrogram, mfcc, audio, sr)
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=target_sr)
    
    # Remove silence
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # Normalize
    audio = librosa.util.normalize(audio)
    
    # Generate spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Generate MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    
    # Segment audio for model input
    segment_samples = int(segment_length * sr)
    overlap_samples = int(overlap * segment_samples)
    step = segment_samples - overlap_samples
    
    segments = []
    for start in range(0, len(audio) - segment_samples + 1, step):
        segment = audio[start:start + segment_samples]
        # Convert to spectrogram
        spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        
        # Resize to 224x224 for ResNet input
        spec_resized = transforms.Resize((224, 224))(torch.tensor(spec_db).unsqueeze(0))
        segments.append(spec_resized)
    
    if not segments:
        # If audio is too short, pad it
        padded_audio = np.pad(audio, (0, segment_samples - len(audio)), 'constant')
        spec = librosa.feature.melspectrogram(y=padded_audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        spec_resized = transforms.Resize((224, 224))(torch.tensor(spec_db).unsqueeze(0))
        segments.append(spec_resized)
    
    return torch.stack(segments), mel_spec_db, mfcc, audio, sr