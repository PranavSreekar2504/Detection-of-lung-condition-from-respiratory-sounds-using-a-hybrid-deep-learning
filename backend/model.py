import torch
import torch.nn as nn
from torchvision import models

class HybridResNetLungDetector(nn.Module):
    """
    Hybrid Deep Learning Model for Lung Disease Detection
    
    Architecture:
    - ResNet50: Feature extraction branch 1 (2048 features)
    - ResNet34: Feature extraction branch 2 (512 features)
    - Fusion: Concatenate both feature vectors
    - Classifier: Dense layers for 6-class classification
    
    Classes:
    0: Normal
    1: Asthma
    2: Pneumonia
    3: COPD
    4: Bronchitis
    5: COVID-19
    """
    
    def __init__(self, num_classes=6, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        
        # ResNet50 - Primary feature extractor
        self.resnet50 = models.resnet50(pretrained=pretrained)
        self.resnet50.fc = nn.Identity()
        # Freeze most layers, fine-tune last block
        for param in self.resnet50.layer1.parameters():
            param.requires_grad = False
        for param in self.resnet50.layer2.parameters():
            param.requires_grad = False
        
        # ResNet34 - Secondary feature extractor
        self.resnet34 = models.resnet34(pretrained=pretrained)
        self.resnet34.fc = nn.Identity()
        # Freeze most layers, fine-tune last block
        for param in self.resnet34.layer1.parameters():
            param.requires_grad = False
        for param in self.resnet34.layer2.parameters():
            param.requires_grad = False
        
        # Feature dimensions after each network
        self.resnet50_features = 2048
        self.resnet34_features = 512
        self.total_features = self.resnet50_features + self.resnet34_features
        
        # Classification head with feature fusion
        self.classifier = nn.Sequential(
            nn.Linear(self.total_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """Forward pass through the hybrid model"""
        features_resnet50 = self.resnet50(x)
        features_resnet34 = self.resnet34(x)
        fused_features = torch.cat([features_resnet50, features_resnet34], dim=1)
        logits = self.classifier(fused_features)
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
    0: "Normal - Healthy respiratory function",
    1: "Asthma - Chronic airway inflammation",
    2: "Pneumonia - Lung infection with fluid accumulation",
    3: "COPD - Chronic obstructive pulmonary disease",
    4: "Bronchitis - Inflammation of bronchi tubes",
    5: "COVID-19 - Novel coronavirus infection"
}

CLASS_RECOMMENDATIONS = {
    0: "Continue regular health monitoring. Maintain healthy lifestyle.",
    1: "Consult pulmonologist. Use appropriate inhalers.",
    2: "Urgent medical evaluation required.",
    3: "Specialist consultation needed.",
    4: "Medical evaluation recommended. Rest and monitor symptoms.",
    5: "Isolate and seek immediate medical attention."
}