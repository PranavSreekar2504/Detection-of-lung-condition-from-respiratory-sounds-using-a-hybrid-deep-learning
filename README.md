# Lung Condition Detection Pipeline

An end-to-end pipeline for detecting lung conditions from respiratory sounds using a hybrid deep learning model.

## Architecture

- **Frontend**: HTML/JavaScript interface for uploading audio files
- **Backend**: FastAPI server with PyTorch model for inference
- **Deployment**: Docker containerized application

## Setup

1. **Train and Save Model**:
   - Run the `ResNet_Model.ipynb` notebook
   - After training, save the model:
   ```python
   torch.save(model.state_dict(), 'backend/model.pth')
   ```

2. **Place Model File**:
   - Copy the trained model file to `backend/model.pth`

3. **Build and Run**:
   ```bash
   docker-compose up --build
   ```

4. **Access Application**:
   - Frontend: http://localhost
   - API: http://localhost/api/predict

## API Usage

POST to `/api/predict` with an audio file:

```bash
curl -X POST -F "file=@audio.wav" http://localhost/api/predict
```

Response:
```json
{
  "prediction": "Normal",
  "confidence": 0.95,
  "class_index": 0
}
```

## Classes

- 0: Normal
- 1: Wheeze
- 2: Crackle
- 3: Both

## Deployment

For production deployment:

1. Build the Docker images
2. Push to a container registry
3. Deploy to cloud platform (AWS ECS, Google Cloud Run, etc.)

## Requirements

- Docker
- Docker Compose
- Trained PyTorch model file