# For deployment to platforms like Heroku, render.com, etc.

# app.py - main application file
from dog_classifier.app import dog_classifier_app
from flask import Flask, render_template, request, jsonify, Response
import requests
import os
import time
import socket
from werkzeug.utils import secure_filename

# Import FastAPI components
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import io
import json
from typing import Any
import numpy as np
import tensorflow as tf
from PIL import Image
import uvicorn
import multiprocessing

import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

# Create upload folder if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

app.register_blueprint(dog_classifier_app, url_prefix="/dog-classifier")

# FastAPI app - We'll run this separately
dog_api = FastAPI(title="Dog Breed Classifier API")

# Add CORS middleware
dog_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FastAPI global variables
model: tf.lite.Interpreter | None = None
label_map: dict[str, str] | None = None
input_details: list[dict[str, Any]] | None = None
output_details: list[dict[str, Any]] | None = None


# FastAPI startup event
@dog_api.on_event("startup")
async def startup_event() -> None:
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
    except Exception as e:
        print(f"Error loading model: {e}")


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


@dog_api.post("/predict/")
async def predict_dog(file: UploadFile = None) -> dict[str, list[dict[str, Any]]]:
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


@dog_api.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Welcome to the Dog Breed Classifier API"}


# Function to run FastAPI in a separate process
def run_fastapi():
    # Use the port from the environment or default to 8000
    port = int(os.environ.get("FASTAPI_PORT", 8000))
    logger.info(f"Starting FastAPI on port {port}")
    try:
        uvicorn.run(dog_api, host="0.0.0.0", port=port)
    except Exception as e:
        logger.error(f"Error starting FastAPI: {e}")


# Determine if we're in production or development
def is_production():
    # Check for common environment variables set in production
    return (
        os.environ.get("PRODUCTION", "").lower() == "true"
        or os.environ.get("ENVIRONMENT", "").lower() == "production"
        or os.environ.get("RENDER", "").lower() == "true"
    )  # Render-specific


# Function to get the host's IP address
def get_host_ip():
    try:
        # This gets the host name of the machine
        hostname = socket.gethostname()
        # This gets the IP address of the machine
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except (socket.gaierror, socket.herror, socket.error) as e:
        logger.warning(f"Could not determine host IP: {e}")
        return "127.0.0.1"  # Fallback to localhost


# Configure the FastAPI URL based on environment
if is_production():
    # In production, FastAPI is running on the same server
    FASTAPI_PORT = int(os.environ.get("FASTAPI_PORT", 8000))
    FASTAPI_URL = f"http://{get_host_ip()}:{FASTAPI_PORT}"
else:
    # In development, use localhost
    FASTAPI_URL = "http://localhost:8000"

print(f"FastAPI URL configured as: {FASTAPI_URL}")


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


# Proxy routes for FastAPI
@app.route("/api/dog/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def proxy_to_fastapi(path):
    # Forward the request to FastAPI
    url = f"{FASTAPI_URL}/{path}"

    # Extract headers, excluding Host which would cause issues
    headers = {
        key: value for (key, value) in request.headers.items() if key.lower() != "host"
    }

    # Forward the request with appropriate method
    try:
        if request.method == "GET":
            resp = requests.get(url, headers=headers, params=request.args, timeout=10)
        elif request.method == "POST":
            # Handle file uploads specially
            if request.files:
                files = {
                    name: (file.filename, file.read())
                    for name, file in request.files.items()
                }
                resp = requests.post(
                    url, headers=headers, files=files, data=request.form, timeout=10
                )
            else:
                resp = requests.post(
                    url, headers=headers, json=request.get_json(), timeout=10
                )
        elif request.method == "PUT":
            resp = requests.put(
                url, headers=headers, json=request.get_json(), timeout=10
            )
        elif request.method == "DELETE":
            resp = requests.delete(url, headers=headers, timeout=10)
        else:
            return Response("Method not allowed", status=405)

        # Return the FastAPI response through Flask
        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("content-type", "text/plain"),
        )
    except requests.exceptions.RequestException as e:
        return jsonify(
            {"error": f"Failed to connect to FastAPI service: {str(e)}"}
        ), 503


@app.route("/api/dog", defaults={"path": ""})
def proxy_root(path):
    return proxy_to_fastapi(path)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/projects")
def projects():
    return render_template("projects.html")


@app.route("/dog_classifier")
def dog_classifier():
    return render_template("dog_classifier.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/android_app")
def android_app():
    return render_template("android_app.html")


@app.route("/this_website")
def this_website():
    return render_template("this_website.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Check if an image was uploaded
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]

    # Check if the user submitted an empty form
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Send the image to your deployed model
        with open(filepath, "rb") as img_file:
            files = {"file": (filename, img_file, "multipart/form-data")}
            response = requests.post(
                "https://dog-breed-model.onrender.com/predict/", files=files
            )

        # Return the prediction results
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify(
                {"error": f"Model API returned error: {response.status_code}"}
            )

    return jsonify({"error": "Invalid file type. Allowed types: png, jpg, jpeg"})


@app.route("/health")
def health_check():
    # Check if we can connect to FastAPI
    try:
        resp = requests.get(f"{FASTAPI_URL}/", timeout=5)
        fastapi_status = resp.status_code == 200
    except requests.RequestException as e:
        logger.warning(f"FastAPI health check failed: {e}")
        fastapi_status = False

    return jsonify({"status": "healthy", "flask": True, "fastapi": fastapi_status})


# For production platforms like Heroku
if __name__ == "__main__":
    # Get the port from the environment
    port = int(os.environ.get("PORT", 10000))

    # Start FastAPI in a separate process if we're not in production
    # For some platforms (like Heroku), you might want to run only Flask
    if not is_production():
        fastapi_process = multiprocessing.Process(target=run_fastapi)
        fastapi_process.daemon = (
            True  # This ensures the process will die when the main process exits
        )
        fastapi_process.start()
        # Give FastAPI time to start up
        time.sleep(1)

    # Run Flask app
    try:
        app.run(host="0.0.0.0", port=port)
    finally:
        # Make sure to clean up the FastAPI process
        if (
            not is_production()
            and "fastapi_process" in locals()
            and fastapi_process
            and fastapi_process.is_alive()
        ):
            fastapi_process.terminate()
            fastapi_process.join()
