import torch
import torch.nn as nn
import numpy as np

class DummyLungDetector(nn.Module):
    """
    Lightweight Dummy Model for Lung Disease Detection
    
    This model simulates the behavior of the HybridResNetLungDetector
    without requiring heavy ResNet architectures or pre-trained weights.
    
    Perfect for testing, development, and demonstration purposes.
    
    Classes:
    0: Normal
    1: Asthma
    2: Pneumonia
    3: COPD
    4: Bronchitis
    5: COVID-19
    """
    
    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        
        # Lightweight feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """Forward pass - produces random but consistent predictions"""
        # Extract features
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        # Classify
        logits = self.classifier(features)
        
        return logits

# Disease class names
CLASS_NAMES = [
    "Normal",
    "Asthma",
    "Pneumonia",
    "COPD",
    "Bronchitis",
    "COVID-19"
]

CLASS_DESCRIPTIONS = {
    0: "Normal - Healthy respiratory function with no abnormal patterns detected.",
    1: "Asthma - Chronic airway inflammation causing breathing difficulty and wheezing.",
    2: "Pneumonia - Lung infection causing fluid/pus accumulation in air sacs.",
    3: "COPD - Chronic obstructive pulmonary disease causing airflow limitation.",
    4: "Bronchitis - Inflammation of bronchi tubes causing persistent cough.",
    5: "COVID-19 - Novel coronavirus infection affecting the respiratory system."
}

CLASS_RECOMMENDATIONS = {
    0: "Continue regular health monitoring. Maintain a healthy lifestyle and monitor for any respiratory changes.",
    1: "Consult with a pulmonologist. Use appropriate inhalers and asthma medications as prescribed.",
    2: "Urgent medical evaluation required. Seek immediate professional medical attention.",
    3: "Schedule a specialist consultation. Discuss treatment options and pulmonary rehabilitation.",
    4: "Medical evaluation recommended. Rest, hydration, and monitor symptoms for 1-2 weeks.",
    5: "Isolate immediately and seek urgent professional medical attention. Follow health authority guidelines."
}
