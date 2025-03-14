# single_process_app.py
import os
import logging
import sys
from flask import Flask, render_template, Blueprint
from fastapi import FastAPI, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import io
import json
from typing import Any
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from werkzeug.utils import secure_filename
from fastapi.staticfiles import StaticFiles

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Create a Flask app for template rendering
flask_app = Flask(__name__)
flask_app.config["UPLOAD_FOLDER"] = "static/uploads"
flask_app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
flask_app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

# Import the dog_classifier_app blueprint
try:
    from dog_classifier.app import dog_classifier_app

    flask_app.register_blueprint(dog_classifier_app, url_prefix="/dog-classifier")
except ImportError as e:
    logger.error(f"Could not import dog_classifier_app: {e}")
    # Create a dummy blueprint if import fails
    dog_classifier_app = Blueprint("dog_classifier", __name__)

    @dog_classifier_app.route("/")
    def dog_classifier_home():
        return "Dog Classifier Blueprint (dummy)"

    flask_app.register_blueprint(dog_classifier_app, url_prefix="/dog-classifier")

# Create FastAPI app
app = FastAPI(title="Combined Flask and FastAPI App")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration values
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model variables
model: tf.lite.Interpreter | None = None
label_map: dict[str, str] | None = None
input_details: list[dict[str, Any]] | None = None
output_details: list[dict[str, Any]] | None = None


# Load model on startup
@app.on_event("startup")
async def startup_event():
    global model, label_map, input_details, output_details
    try:
        # Load the TF Lite model
        interpreter = tf.lite.Interpreter(model_path="dog_breed_model_quantized.tflite")
        interpreter.allocate_tensors()
        model = interpreter

        # Get input and output tensors
        input_details = model.get_input_details()
        output_details = model.get_output_details()

        # Load the label map
        with open("label_map.json") as f:
            label_map = json.load(f)

        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(
    image: Image.Image, target_size: tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Preprocess the image for model prediction."""
    # Resize image
    image = image.resize(target_size)

    # Convert to array and normalize
    image_array = np.array(image)
    # Handle grayscale images
    if len(image_array.shape) == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)
    elif image_array.shape[2] == 4:
        # Remove alpha channel
        image_array = image_array[:, :, :3]

    # Normalize to [0,1]
    image_array = image_array.astype(np.float32) / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


# Helper function to render templates with Flask
def render_flask_template(template_name, **context):
    with flask_app.app_context():
        return render_template(template_name, **context)


# FastAPI Endpoints


@app.get("/api/dog")
def read_root():
    return {"message": "Welcome to the Dog Breed Classifier API"}


@app.post("/api/dog/predict")
async def predict_dog(file: UploadFile = None):
    """
    Predict the dog breed from an uploaded image.
    """
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")
    if (
        model is None
        or input_details is None
        or output_details is None
        or label_map is None
    ):
        raise HTTPException(status_code=500, detail="Model not initialized")

    # Read and process the image
    image_content = await file.read()
    image = Image.open(io.BytesIO(image_content))
    processed_image = preprocess_image(image)

    # Make prediction using TF Lite
    model.set_tensor(input_details[0]["index"], processed_image)
    model.invoke()
    predictions = model.get_tensor(output_details[0]["index"])

    # Get top 3 predictions
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [
        {"breed": label_map[str(idx)], "confidence": float(predictions[0][idx])}
        for idx in top_3_indices
    ]

    return {"predictions": top_3_predictions}


# Flask-like routes for FastAPI


@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        html_content = render_flask_template("index.html")
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error rendering home template: {e}")
        return HTMLResponse(
            content=f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>"
        )


@app.get("/projects", response_class=HTMLResponse)
async def projects():
    return HTMLResponse(content=render_flask_template("projects.html"))


@app.get("/dog_classifier", response_class=HTMLResponse)
async def dog_classifier():
    return HTMLResponse(content=render_flask_template("dog_classifier.html"))


@app.get("/about", response_class=HTMLResponse)
async def about():
    return HTMLResponse(content=render_flask_template("about.html"))


@app.get("/android_app", response_class=HTMLResponse)
async def android_app():
    return HTMLResponse(content=render_flask_template("android_app.html"))


@app.get("/this_website", response_class=HTMLResponse)
async def this_website():
    return HTMLResponse(content=render_flask_template("this_website.html"))


@app.post("/predict")
async def predict(request: Request):
    # Get the form data
    form = await request.form()

    # Check if an image was uploaded
    if "file" not in form:
        return JSONResponse({"error": "No file part"})

    file = form["file"]

    # Check if the user submitted an empty form
    if file.filename == "":
        return JSONResponse({"error": "No selected file"})

    if file and allowed_file(file.filename):
        # Save the uploaded file
        contents = await file.read()
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        with open(filepath, "wb") as f:
            f.write(contents)

        # Send the image to your deployed model
        with open(filepath, "rb") as img_file:
            files = {"file": (filename, img_file, "multipart/form-data")}
            response = requests.post(
                "https://dog-breed-model.onrender.com/predict/", files=files
            )

        # Return the prediction results
        if response.status_code == 200:
            return JSONResponse(response.json())
        else:
            return JSONResponse(
                {"error": f"Model API returned error: {response.status_code}"}
            )

    return JSONResponse({"error": "Invalid file type. Allowed types: png, jpg, jpeg"})


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "flask_routes": True,
        "fastapi_routes": True,
        "model_loaded": model is not None,
    }


# Make it compatible with render.com and other hosting providers
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting application on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
