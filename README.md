# Personal Portfolio with Dog Breed Classifier API

This project is a personal portfolio website integrated with a Dog Breed Classifier API. The application combines a Flask web application with FastAPI endpoints in a single-process architecture for reliable deployment.

## Features

- Personal portfolio website with responsive design
- Dog breed classifier that can identify breeds from uploaded images
- RESTful API for programmatic access to the classifier
- Health monitoring endpoints
- Error handling and logging

## Architecture

The application uses a hybrid approach that combines:

- **Flask**: For template rendering and the main website
- **FastAPI**: For the dog breed classifier API
- **TensorFlow Lite**: For running the dog breed classification model

All components run in a single process, making it easier to deploy on platforms like Render.com.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/projects` | GET | Projects overview |
| `/about` | GET | About page |
| `/api/dog` | GET | API root with welcome message |
| `/api/dog/predict` | POST | Dog breed prediction endpoint (FastAPI) |
| `/predict` | POST | Alternative prediction endpoint (FastAPI) |
| `/health` | GET | Health check endpoint |

## Dog Breed Classifier

The dog breed classifier uses a TensorFlow Lite model to identify dog breeds from uploaded images. The model is trained to recognize numerous dog breeds with high accuracy.

### Example Usage

```python
import requests

# Prepare the image file
with open("your_dog_image.jpg", "rb") as image_file:
    files = {"file": ("your_dog_image.jpg", image_file, "image/jpeg")}

    # Send the request
    response = requests.post("https://prashand.nl/api/dog/predict", files=files)

    # Process the results
    if response.status_code == 200:
        predictions = response.json()["predictions"]
        for prediction in predictions:
            print(f"{prediction['breed']}: {prediction['confidence'] * 100:.2f}%")
```

## Important Implementation Details

- The application uses a single-process architecture where FastAPI handles all HTTP requests
- Flask is used internally for template rendering
- Prediction is done directly using the TensorFlow Lite model without making additional HTTP requests
- Both `/api/dog/predict` and `/predict` endpoints use the same underlying model for predictions

## Development Setup

### Prerequisites

- Python 3.9+
- pip
- virtualenv (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Make sure you have the model files:
   - `dog_breed_model_quantized.tflite`
   - `label_map.json`

5. Run the application:
   ```bash
   python single_process_app.py
   ```

The application will be available at http://localhost:10000

## Deployment

The application is designed to be easily deployed on Render.com or similar platforms.

### Deployment on Render.com

1. Connect your GitHub repository to Render.com
2. Create a new Web Service
3. Configure the service:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `./start.sh`
4. Add environment variables:
   - `PRODUCTION`: `true`
   - `PORT`: `10000` (or let Render set it automatically)

## Project Structure

```
├── single_process_app.py     # Main application file
├── start.sh                  # Startup script for deployment
├── requirements.txt          # Python dependencies
├── static/                   # Static files (CSS, JS, images)
│   ├── css/
│   ├── js/
│   └── uploads/              # Uploaded images directory
├── templates/                # HTML templates
│   ├── index.html
│   ├── projects.html
│   └── ...
├── dog_classifier/           # Dog classifier module
│   └── app.py                # Dog classifier blueprint
├── dog_breed_model_quantized.tflite  # TensorFlow Lite model
└── label_map.json            # Mapping of model outputs to breed names
```

## License

[MIT License](LICENSE)

## Acknowledgments

- TensorFlow for the model architecture
- Stanford Dogs Dataset for training data
- Flask and FastAPI for the web frameworks
