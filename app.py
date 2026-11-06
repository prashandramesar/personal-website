"""Portfolio site and dog breed classifier, served by a single FastAPI app."""

import io
import json
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from ai_edge_litert.interpreter import Interpreter
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from flask import Flask, render_template
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "dog_breed_model_quantized.tflite"
LABEL_MAP_PATH = BASE_DIR / "label_map.json"

MAX_UPLOAD_BYTES = 16 * 1024 * 1024
# Allows for multipart boundaries and headers around a maximum-size file.
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 64 * 1024
# Compressed images can be small on disk but huge once decoded.
MAX_IMAGE_PIXELS = 50_000_000
# Checked against the file contents, as filenames can claim anything.
ALLOWED_IMAGE_FORMATS = ("JPEG", "PNG")
INVALID_IMAGE_MESSAGE = "Invalid image. Allowed types: png, jpg, jpeg"
TOO_LARGE_MESSAGE = "File too large. The maximum size is 16 MB"

# The Android app is not a browser, so CORS does not apply to it.
ALLOWED_ORIGINS = ["https://prashand.nl", "https://www.prashand.nl"]

# Flask only renders Jinja templates. FastAPI handles every request.
flask_app = Flask(__name__)


@flask_app.context_processor
def inject_current_year() -> dict[str, int]:
    return {"current_year": date.today().year}


model: Interpreter | None = None
label_map: dict[str, str] | None = None
input_details: list[dict[str, Any]] | None = None
output_details: list[dict[str, Any]] | None = None


class UploadError(Exception):
    """An upload the classifier can't process, with the status to report."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class RequestTooLarge(Exception):
    """Raised while reading a request body that goes over the limit."""


class MaxBodySizeMiddleware:
    """Reject oversized requests before their body is buffered.

    Counting the bytes as well as checking Content-Length also covers chunked
    requests, which don't declare a length.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope["headers"])
        declared_length = headers.get(b"content-length", b"")
        too_large = JSONResponse({"error": TOO_LARGE_MESSAGE}, status_code=413)
        if declared_length.isdigit() and int(declared_length) > self.max_bytes:
            await too_large(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestTooLarge:
            await too_large(scope, receive, send)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    load_model()
    yield


def load_model() -> None:
    global model, label_map, input_details, output_details
    try:
        interpreter = Interpreter(model_path=str(MODEL_PATH))
        interpreter.allocate_tensors()
        with open(LABEL_MAP_PATH, encoding="utf-8") as f:
            labels = json.load(f)
    except (OSError, ValueError, RuntimeError):
        # The site can still serve its pages, and /health reports the failure.
        logger.exception("Error loading model")
        return

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    label_map = labels
    model = interpreter
    logger.info("Model loaded successfully")


app = FastAPI(title="Prashand Ramesar Portfolio", lifespan=lifespan)
app.add_middleware(MaxBodySizeMiddleware, max_bytes=MAX_REQUEST_BYTES)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


def preprocess_image(
    image: Image.Image, target_size: tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Convert an image to the model's input: RGB, 224x224, scaled to [0, 1]."""
    # Convert before resizing, because palette images can't be interpolated.
    image = ImageOps.exif_transpose(image).convert("RGB").resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def load_upload(contents: bytes) -> np.ndarray:
    """Validate uploaded bytes and return them as model input."""
    if len(contents) > MAX_UPLOAD_BYTES:
        raise UploadError(413, TOO_LARGE_MESSAGE)

    try:
        image = Image.open(io.BytesIO(contents), formats=ALLOWED_IMAGE_FORMATS)
        width, height = image.size
        if width * height > MAX_IMAGE_PIXELS:
            raise UploadError(413, "Image dimensions are too large")
        return preprocess_image(image)
    except Image.DecompressionBombError as e:
        raise UploadError(413, "Image dimensions are too large") from e
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
        raise UploadError(400, INVALID_IMAGE_MESSAGE) from e


def format_breed_name(label: str) -> str:
    """Turn 'n02097047-miniature_schnauzer' into 'Miniature Schnauzer'."""
    words = label.split("-", 1)[-1].split("_")
    return " ".join(word[0].upper() + word[1:] for word in words)


def classify(contents: bytes) -> list[dict[str, str | float]]:
    """Return the top 3 breeds for an uploaded image.

    Uploads stay in memory, so visitors' photos are never written to disk.
    """
    if (
        model is None
        or input_details is None
        or output_details is None
        or label_map is None
    ):
        raise UploadError(503, "Model not initialized")

    processed_image = load_upload(contents)

    # The interpreter holds state between these calls and isn't thread-safe.
    # They run on the event loop with no await between them, so requests can't
    # interleave. Add a lock if this ever moves to a thread pool.
    model.set_tensor(input_details[0]["index"], processed_image)
    model.invoke()
    predictions = model.get_tensor(output_details[0]["index"])[0]

    top_3_indices = np.argsort(predictions)[-3:][::-1]
    return [
        {
            "breed": label_map[str(idx)],
            "display_name": format_breed_name(label_map[str(idx)]),
            "confidence": float(predictions[idx]),
        }
        for idx in top_3_indices
    ]


def render_page(template_name: str) -> HTMLResponse:
    with flask_app.app_context():
        return HTMLResponse(content=render_template(template_name))


@app.get("/api/dog")
def read_root() -> dict[str, str]:
    return {"message": "Welcome to the Dog Breed Classifier API"}


@app.post("/api/dog/predict")
async def predict_dog(
    file: UploadFile | None = File(default=None),
) -> dict[str, list[dict[str, str | float]]]:
    """Predict the dog breed from an uploaded image."""
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")
    try:
        return {"predictions": classify(await file.read())}
    except UploadError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message) from e


@app.post("/predict")
async def predict(request: Request) -> JSONResponse:
    """Predict the breed for the website form and the Android app.

    Released Android builds call this route and read `breed` and `confidence`,
    so don't remove or rename those fields.
    """
    form = await request.form()
    file = form.get("file")

    if not isinstance(file, StarletteUploadFile):
        return JSONResponse({"error": "No file part"}, status_code=400)
    if not file.filename:
        return JSONResponse({"error": "No selected file"}, status_code=400)

    try:
        return JSONResponse({"predictions": classify(await file.read())})
    except UploadError as e:
        return JSONResponse({"error": e.message}, status_code=e.status_code)


@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    return render_page("index.html")


@app.get("/projects", response_class=HTMLResponse)
async def projects() -> HTMLResponse:
    return render_page("projects.html")


@app.get("/dog_classifier", response_class=HTMLResponse)
async def dog_classifier() -> HTMLResponse:
    return render_page("dog_classifier.html")


@app.get("/about", response_class=HTMLResponse)
async def about() -> HTMLResponse:
    return render_page("about.html")


@app.get("/android_app", response_class=HTMLResponse)
async def android_app() -> HTMLResponse:
    return render_page("android_app.html")


@app.get("/this_website", response_class=HTMLResponse)
async def this_website() -> HTMLResponse:
    return render_page("this_website.html")


@app.get("/health")
async def health_check() -> JSONResponse:
    model_loaded = model is not None
    return JSONResponse(
        {
            "status": "healthy" if model_loaded else "unhealthy",
            "model_loaded": model_loaded,
        },
        status_code=200 if model_loaded else 503,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting application on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
