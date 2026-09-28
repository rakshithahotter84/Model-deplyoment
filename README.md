# ML Model Deployment API

A clean, containerized machine-learning inference API built with **FastAPI** and **Docker**.

This repository is structured for an ML deployment assignment:

- API endpoint for model inference
- Dockerfile for containerization
- Example request and response
- Local run instructions
- Swagger/OpenAPI documentation
- Health-check endpoint

## Project Structure

```text
ml-model-api/
├── app.py
├── model.pkl
├── train_model.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── sample_request.json
└── README.md
```

## Model

The included `model.pkl` is a demo **Random Forest classifier trained on the Iris dataset** so the repository can be run immediately.

If your internship task uses a different trained model, replace `model.pkl` and update the input format in `app.py` to match your model.

## Run Locally Without Docker

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**Linux/macOS**
```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the API

```bash
uvicorn app:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

## API Endpoints

### GET `/`

Returns basic API information.

### GET `/health`

Checks whether the model was loaded successfully.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### POST `/predict`

Accepts four numeric Iris features.

Request:

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Example response:

```json
{
  "prediction": 0,
  "probabilities": [0.99, 0.01, 0.0]
}
```

## Test With cURL

```bash
curl -X POST "http://localhost:8000/predict"   -H "Content-Type: application/json"   -d "{"features":[5.1,3.5,1.4,0.2]}"
```

For Linux/macOS:

```bash
curl -X POST "http://localhost:8000/predict"   -H "Content-Type: application/json"   -d '{"features":[5.1,3.5,1.4,0.2]}'
```

## Run With Docker

### 1. Build the image

```bash
docker build -t ml-model-api .
```

### 2. Run the container

```bash
docker run --rm -p 8000:8000 ml-model-api
```

### 3. Test the API

Open:

```text
http://localhost:8000/docs
```

Or run:

```bash
curl -X POST "http://localhost:8000/predict"   -H "Content-Type: application/json"   -d '{"features":[5.1,3.5,1.4,0.2]}'
```

## Recreating the Demo Model

If `model.pkl` needs to be regenerated:

```bash
python train_model.py
```

This creates a new `model.pkl` using the Iris dataset.

## GitHub Submission

Create a new GitHub repository and upload all files in this folder.

Suggested repository name:

```text
ml-model-deployment-api
```

Suggested repository description:

```text
Containerized machine-learning inference API using FastAPI and Docker.
```

## Assignment Deliverables

| Requirement | Included |
|---|---|
| Model inference endpoint | Yes |
| FastAPI implementation | Yes |
| Dockerfile | Yes |
| Local run instructions | Yes |
| Example request | Yes |
| Example response | Yes |
| GitHub-ready structure | Yes |
| Short demo via cURL | Yes |
| Health check | Yes |

## Notes for Custom Models

For your actual trained model, make sure the following are consistent:

1. The feature order in the API matches training.
2. Any preprocessing/scaling used during training is included.
3. The saved model and preprocessing pipeline are loaded together when required.
4. The request schema matches the model's expected input.
5. The README is updated with the actual model and example prediction.

## License

This project is provided for educational and internship submission purposes.
