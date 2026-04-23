import torch
from src.model import HybridResNetLungDetector
from src.preprocess import preprocess_audio
import numpy as np

def predict_lung_condition(audio_path):
    """
    Predict lung condition from audio file.
    
    Args:
        audio_path (str): Path to audio file
    
    Returns:
        dict: Prediction results with class, confidence, and probabilities
    """
    # Load model
    model = HybridResNetLungDetector(num_classes=6)
    model.load_state_dict(torch.load('models/best_model.pth', map_location='cpu'))
    model.eval()
    
    # Preprocess audio
    segments, _, _, _, _ = preprocess_audio(audio_path)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(segments)
        probabilities = torch.softmax(outputs, dim=1)
        avg_probabilities = probabilities.mean(dim=0)
        predicted_class_idx = torch.argmax(avg_probabilities).item()
    
    class_names = model.CLASS_NAMES
    result = {
        'predicted_class': class_names[predicted_class_idx],
        'confidence': float(avg_probabilities[predicted_class_idx]),
        'probabilities': {class_names[i]: float(prob) for i, prob in enumerate(avg_probabilities)},
        'description': model.CLASS_DESCRIPTIONS[class_names[predicted_class_idx]],
        'recommendation': model.CLASS_RECOMMENDATIONS[class_names[predicted_class_idx]]
    }
    
    return result

if __name__ == "__main__":
    # Example usage
    result = predict_lung_condition("path/to/audio.wav")
    print(result)