from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import numpy as np
from model import HybridResNetLungDetector, CLASS_NAMES, CLASS_DESCRIPTIONS, CLASS_RECOMMENDATIONS
from preprocess import preprocess_audio

app = FastAPI(
    title="RespiCare - Lung Disease Detection API",
    description="AI-powered respiratory disease detection from audio",
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

# Load real hybrid model
BASE_DIR = os.path.dirname(__file__)
model_name = os.getenv("MODEL_PATH", "cascade_hybrid_model.pth")
model_path = os.path.join(BASE_DIR, model_name)

# support common alternate filenames
alt_paths = [
    os.path.join(BASE_DIR, "cascade_hybrid_model.pth"),
    os.path.join(BASE_DIR, "model.pth"),
    os.path.join(BASE_DIR, "backend", "model.pth")
]
if not os.path.isfile(model_path):
    for alt in alt_paths:
        if os.path.isfile(alt):
            model_path = alt
            break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Using device: {device}")
print(f"ℹ Attempting to load model from: {model_path}")

model = HybridResNetLungDetector(num_classes=len(CLASS_NAMES), pretrained=False)
model = model.to(device)
model_loaded = False

if os.path.isfile(model_path):
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model_state_dict = checkpoint["state_dict"]
        else:
            model_state_dict = checkpoint
        model.load_state_dict(model_state_dict)
        model_loaded = True
        print("✓ Loaded model weights successfully")
    except Exception as load_error:
        print(f"✗ Failed to load model weights: {load_error}")
        model_loaded = False
else:
    print(f"⚠ Warning: model file not found at {model_path}. Prediction service will be disabled until weights are added.")

model.eval()
print("✓ HybridResNet model ready")

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
        
        if not model_loaded:
            return JSONResponse(
                content={"error": "Model weights not loaded. Please place the trained .pth file in backend/ and restart the server."},
                status_code=503
            )

        # Preprocess audio
        img, mel_spec_db, mfcc, y, sr = preprocess_audio(audio_bytes)

        # Convert input to tensor and evaluate
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_idx])

        # Get predictions for all classes
        all_predictions = {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(CLASS_NAMES))
        }
        if "Normal" not in all_predictions:
            all_predictions["Normal"] = 0.0

        prediction = CLASS_NAMES[pred_idx]
        description = CLASS_DESCRIPTIONS.get(pred_idx, "Unknown condition")
        recommendation = CLASS_RECOMMENDATIONS.get(pred_idx, "Consult healthcare professional")

        # Create severity level
        severity_map = {
            0: "Normal",     # Normal
            1: "Moderate",   # Asthma
            2: "High",       # Pneumonia
            3: "Moderate",   # COPD
            4: "Moderate",   # Bronchitis
            5: "Critical"    # COVID-19
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

            if not model_loaded:
                results.append({
                    "filename": file.filename,
                    "error": "Model weights not loaded. Please add the trained .pth file to backend/ and restart the server."
                })
                continue

            # Preprocess
            img, mel_spec_db, mfcc, y, sr = preprocess_audio(audio_bytes)
            
            # Predict
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probabilities))
                confidence = float(probabilities[pred_idx])
                all_predictions = {
                    CLASS_NAMES[i]: float(probabilities[i])
                    for i in range(len(CLASS_NAMES))
                }
            
            results.append({
                "filename": file.filename,
                "prediction": CLASS_NAMES[pred_idx],
                "confidence": confidence,
                "class_index": pred_idx,
                "all_predictions": all_predictions
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results, "total_files": len(files)}