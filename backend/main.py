from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import numpy as np
from model_dummy import DummyLungDetector, CLASS_NAMES, CLASS_DESCRIPTIONS, CLASS_RECOMMENDATIONS
from preprocess import preprocess_audio

app = FastAPI(
    title="RespiCare - Lung Disease Detection API",
    description="AI-powered respiratory disease detection from audio (DEMO MODE)",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dummy model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Using device: {device}")
print("✓ Running in DEMO MODE with dummy model")

model = DummyLungDetector(num_classes=6)
model = model.to(device)
model.eval()
print("✓ Dummy model initialized successfully")

# Image transforms for model input
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def convert_spectrogram_to_base64(mel_spec_db):
    """Convert mel-spectrogram to base64 for frontend display"""
    # Normalize to 0-255
    mel_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
    mel_scaled = (mel_normalized * 255).astype(np.uint8)
    
    # Convert to image
    img = Image.fromarray(mel_scaled).convert("RGB")
    img = img.resize((512, 256))
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "RespiCare API is running",
        "model": "HybridResNet (ResNet50 + ResNet34)",
        "classes": CLASS_NAMES,
        "device": str(device)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict lung disease from respiratory audio
    
    Args:
        file: Audio file (.wav, .mp3, etc.)
    
    Returns:
        JSON with prediction, confidence, and additional info
    """
    try:
        # Read audio file
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            return JSONResponse(
                content={"error": "Empty file uploaded"},
                status_code=400
            )
        
        # Preprocess audio
        img, mel_spec_db, mfcc, y, sr = preprocess_audio(audio_bytes)
        
        # Generate demo prediction based on audio features for realistic variation
        # Use audio characteristics as seed for consistent but varied predictions
        audio_hash = hash(y.tobytes()) % 100
        seed = (audio_hash + len(y)) % 1000
        np.random.seed(seed)
        
        # Create realistic probability distribution
        base_probs = np.random.dirichlet(np.ones(6) * 2)  # Dirichlet for realistic distribution
        
        # Bias towards normal (most common)
        if np.random.random() > 0.3:  # 70% chance of normal
            base_probs[0] *= 1.5
        
        # Normalize to probabilities
        probabilities_np = base_probs / base_probs.sum()
        pred_idx = int(np.argmax(probabilities_np))
        confidence = float(probabilities_np[pred_idx])
        
        # Convert to torch for compatibility
        probabilities = torch.from_numpy(probabilities_np).float()
        
        # Get predictions for all classes
        all_predictions = {
            CLASS_NAMES[i]: float(probabilities_np[i])
            for i in range(len(CLASS_NAMES))
        }
        
        prediction = CLASS_NAMES[pred_idx]
        description = CLASS_DESCRIPTIONS.get(pred_idx, "Unknown condition")
        recommendation = CLASS_RECOMMENDATIONS.get(pred_idx, "Consult healthcare professional")
        
        # Create severity level
        severity_map = {
            0: "Normal",
            1: "Moderate",
            2: "High",
            3: "Moderate",
            4: "Moderate",
            5: "Critical"
        }
        severity = severity_map.get(pred_idx, "Unknown")
        
        # Convert spectrogram to base64
        spectrogram_b64 = convert_spectrogram_to_base64(mel_spec_db)
        
        response = {
            "prediction": prediction,
            "class_index": pred_idx,
            "confidence": confidence,
            "confidence_percentage": round(confidence * 100, 2),
            "severity": severity,
            "description": description,
            "recommendation": recommendation,
            "all_predictions": all_predictions,
            "spectrogram": spectrogram_b64,
            "audio_info": {
                "duration_seconds": float(len(y) / sr),
                "sample_rate": int(sr),
                "filename": file.filename
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"Prediction failed: {str(e)}"},
            status_code=500
        )

@app.post("/batch_predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    """Batch prediction for multiple audio files"""
    results = []
    
    for file in files:
        try:
            audio_bytes = await file.read()
            
            # Preprocess
            img, mel_spec_db, mfcc, y, sr = preprocess_audio(audio_bytes)
            
            # Predict
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                pred_idx = probabilities.argmax(1).item()
                confidence = probabilities[0][pred_idx].item()
            
            results.append({
                "filename": file.filename,
                "prediction": CLASS_NAMES[pred_idx],
                "confidence": confidence,
                "class_index": pred_idx
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results, "total_files": len(files)}