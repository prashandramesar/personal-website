# Personal Portfolio with Dog Breed Classifier API

This project is a personal portfolio website with a built-in dog breed classifier and API, running as a single FastAPI application.

## Features

- Personal portfolio website with responsive design
- Dog breed classifier that identifies 120 breeds from uploaded images
- RESTful API for programmatic access to the classifier
- Health check endpoint for monitoring

## Architecture

- **FastAPI**: Handles all HTTP requests, including pages, static files and the API
- **Flask**: Renders the Jinja templates only
- **LiteRT**: Runs the quantized MobileNetV2 model in TensorFlow Lite format

Predictions call the model directly in the same process. Uploaded images are processed in memory and are never saved.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/projects` | GET | Projects overview |
| `/dog_classifier` | GET | Classifier demo page |
| `/android_app` | GET | Android app project page |
| `/this_website` | GET | Website project page |
| `/about` | GET | About page |
| `/api/dog` | GET | API root with welcome message |
| `/api/dog/predict` | POST | Dog breed prediction |
| `/predict` | POST | Dog breed prediction for the website and Android app |
| `/health` | GET | Health check, returns 503 if the model failed to load |

## Dog Breed Classifier

Prediction endpoints accept a multipart `file` field containing a JPEG or PNG of up to 16 MB, and return the top 3 breeds:

```json
{
  "predictions": [
    {
      "breed": "n02099601-golden_retriever",
      "display_name": "Golden Retriever",
      "confidence": 0.997
    }
  ]
}
```

Invalid images return 400, oversized files return 413, and 503 means the model isn't available.

The model always picks one of its 120 breeds, even for a photo with no dog in it, so treat low confidence as "not sure". The website shows a warning below 30%.

The training code is in the [dog-breed-model](https://github.com/prashandramesar/dog-breed-model) repository.

### Example Usage

```python
import requests

with open("your_dog_image.jpg", "rb") as image_file:
    files = {"file": ("your_dog_image.jpg", image_file, "image/jpeg")}
    response = requests.post("https://prashand.nl/api/dog/predict", files=files)

if response.status_code == 200:
    for prediction in response.json()["predictions"]:
        print(f"{prediction['display_name']}: {prediction['confidence'] * 100:.2f}%")
```

## Development Setup

### Prerequisites

- Python 3.11
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/prashandramesar/personal-website.git
   cd personal-website
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

The application will be available at http://localhost:10000

### Checks

These also run in CI on pull requests and on `main`:

```bash
ruff check .
ruff format --check .
mypy .
pytest
```

## Deployment

The site is deployed on Render using `render.yaml`:

- Build Command: `pip install -r requirements.txt`
- Start Command: `./start.sh`
- Health Check Path: `/health`

`requirements.txt` holds the runtime dependencies only. Render sets `PORT` automatically.

## Project Structure

```
├── app.py                    # Main application file
├── start.sh                  # Startup script for deployment
├── render.yaml               # Render service configuration
├── requirements.txt          # Runtime dependencies
├── requirements-dev.txt      # Runtime dependencies plus dev tools
├── static/                   # Images and the CV
├── templates/                # HTML templates, each extending base.html
├── tests/                    # Test suite and sample images
├── dog_breed_model_quantized.tflite  # TensorFlow Lite model
└── label_map.json            # Mapping of model outputs to breed names
```

## Acknowledgments

- TensorFlow for the model tooling
- Stanford Dogs Dataset for training data
- Flask and FastAPI for the web frameworks
