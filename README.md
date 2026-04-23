# RespiCare: Hybrid Deep Learning for Lung Condition Detection

RespiCare is a state-of-the-art diagnostic platform that uses a hybrid ResNet architecture (ResNet50 + ResNet34) to detect various respiratory conditions from audio recordings.

## Features
- **Hybrid Model:** Combines features from ResNet50 and ResNet34 for superior diagnostic accuracy.
- **Multi-Class Detection:** Detects Normal, Asthma, Pneumonia, COPD, Bronchitis, and COVID-19.
- **FastAPI Backend:** High-performance inference engine with audio chunking and specialized aggregation.
- **Modern UI:** Glassmorphic, responsive web interface for real-time analysis.

## Project Structure
- `backend/`: FastAPI application and inference logic.
- `frontend/`: Web interface (HTML/CSS/JS).
- `src/`: Training pipeline and dataset utilities.
- `models/`: Trained model checkpoints.
- `data/`: (Local only) ICBHI and Coswara dataset files.

## Getting Started

### Prerequisites
- Python 3.9+
- librosa, torch, torchaudio, fastapi, uvicorn

### Installation
1. Clone the repository.
2. Create a virtual environment: `python -m venv .venv`.
3. Activate the environment: `source .venv/bin/activate`.
4. Install dependencies: `pip install -r requirements.txt`.

### Running the Application
1. Start the backend:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```
2. Open `frontend/index.html` in your browser.

## Training
To retrain the model, use the scripts in the `src/` directory:
```bash
python src/train.py
```

## Dataset Credits
- ICBHI 2017 Challenge Dataset
- Coswara COVID-19 Dataset