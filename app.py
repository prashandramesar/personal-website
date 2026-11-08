"""Portfolio site and dog breed classifier, served by a single FastAPI app."""

import asyncio
import io
import json
import logging
import os
import re
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import uvicorn
from ai_edge_litert.interpreter import Interpreter
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from flask import Flask, render_template
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from recipes import backup as recipe_backup
from recipes.routes import create_router as create_recipe_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "dog_breed_model_quantized.tflite"
LABEL_MAP_PATH = BASE_DIR / "label_map.json"
# Images dropped in here appear on the recipe tool's page. The filename sets
# the order and caption, so "01-review-page.png" becomes "Review page".
SCREENSHOT_DIR = BASE_DIR / "static" / "img" / "recipe_tool"
SCREENSHOT_TYPES = {".png", ".jpg", ".jpeg", ".webp"}

MAX_UPLOAD_BYTES = 16 * 1024 * 1024
# Allows for multipart boundaries and headers around a maximum-size file.
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 64 * 1024
# Compressed images can be small on disk but huge once decoded, and the Render
# instance has 512 MB. JPEGs can be decoded at reduced size, but PNGs can't.
MAX_JPEG_PIXELS = 50_000_000
MAX_PNG_PIXELS = 25_000_000
JPEG_DECODE_SIZE = (448, 448)
# Checked against the file contents, as filenames can claim anything.
ALLOWED_IMAGE_FORMATS = ("JPEG", "PNG")
INVALID_IMAGE_MESSAGE = "Invalid image. Allowed types: png, jpg, jpeg"
TOO_LARGE_MESSAGE = "File too large. The maximum size is 16 MB"

# The Android app is not a browser, so CORS does not apply to it.
ALLOWED_ORIGINS = ["https://prashand.nl", "https://www.prashand.nl"]

SITE_URL = "https://www.prashand.nl"
PAGES = {
    "/": "index.html",
    "/kaai": "kaai.html",
    "/projects": "projects.html",
    "/dog_classifier": "dog_classifier.html",
    "/android_app": "android_app.html",
    "/recipe_tool": "recipe_tool.html",
    "/this_website": "this_website.html",
    "/about": "about.html",
}

# Flask only renders Jinja templates. FastAPI handles every request.
flask_app = Flask(__name__)


def find_screenshots(folder: Path = SCREENSHOT_DIR) -> list[dict[str, str]]:
    """List a folder's images in filename order, with captions from their names."""
    if not folder.is_dir():
        return []
    screenshots = []
    for path in sorted(folder.iterdir(), key=lambda item: item.name.lower()):
        if path.suffix.lower() not in SCREENSHOT_TYPES:
            continue
        words = re.sub(r"[-_]+", " ", re.sub(r"^\d+[-_ ]*", "", path.stem)).strip()
        screenshots.append(
            {
                "url": f"/static/img/recipe_tool/{quote(path.name)}",
                "caption": words[:1].upper() + words[1:],
            }
        )
    return screenshots


# Images only change with a deploy, so reading the folder once is enough.
RECIPE_SCREENSHOTS = find_screenshots()


@flask_app.context_processor
def inject_template_values() -> dict[str, Any]:
    return {
        "current_year": date.today().year,
        "recipe_screenshots": RECIPE_SCREENSHOTS,
    }


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
    backup_config = recipe_backup.backup_config()
    db_path = os.environ.get("RECIPES_DB_PATH")
    backup_task = (
        asyncio.create_task(recipe_backup.run_forever(Path(db_path), backup_config))
        if backup_config and db_path
        else None
    )
    yield
    if backup_task:
        backup_task.cancel()


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
    # Each of these steps avoids holding another full-size copy in memory.
    image.draft("RGB", JPEG_DECODE_SIZE)
    ImageOps.exif_transpose(image, in_place=True)
    # Convert before resizing, because palette images can't be interpolated.
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_array = np.array(image.resize(target_size), dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def load_upload(contents: bytes) -> np.ndarray:
    """Validate uploaded bytes and return them as model input."""
    if len(contents) > MAX_UPLOAD_BYTES:
        raise UploadError(413, TOO_LARGE_MESSAGE)

    try:
        image = Image.open(io.BytesIO(contents), formats=ALLOWED_IMAGE_FORMATS)
        width, height = image.size
        max_pixels = MAX_JPEG_PIXELS if image.format == "JPEG" else MAX_PNG_PIXELS
        if width * height > max_pixels:
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


def render_page(
    template_name: str,
    path: str | None = None,
    status_code: int = 200,
    **context: Any,
) -> HTMLResponse:
    canonical_url = f"{SITE_URL}{path}" if path else None
    with flask_app.app_context():
        html = render_template(
            template_name, canonical_url=canonical_url, site_url=SITE_URL, **context
        )
    return HTMLResponse(content=html, status_code=status_code)


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


def add_page(path: str, template_name: str) -> None:
    async def page() -> HTMLResponse:
        return render_page(template_name, path)

    app.add_api_route(
        path,
        page,
        methods=["GET"],
        response_class=HTMLResponse,
        name=template_name.removesuffix(".html"),
    )


for page_path, page_template in PAGES.items():
    add_page(page_path, page_template)

app.include_router(create_recipe_router(render_page))


@app.exception_handler(404)
async def not_found(request: Request, exc: Exception) -> Response:
    """Show visitors a page, and keep JSON errors for API clients."""
    if request.method == "GET" and "text/html" in request.headers.get("accept", ""):
        return render_page("404.html", status_code=404)
    return JSONResponse({"detail": "Not Found"}, status_code=404)


@app.get("/robots.txt", response_class=PlainTextResponse)
async def robots() -> str:
    return f"User-agent: *\nAllow: /\nSitemap: {SITE_URL}/sitemap.xml\n"


@app.get("/sitemap.xml")
async def sitemap() -> Response:
    urls = "".join(f"<url><loc>{SITE_URL}{path}</loc></url>" for path in PAGES)
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>'
    )
    return Response(content=body, media_type="application/xml")


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
