# dog_classifier/app.py
from flask import Blueprint, render_template, request, jsonify
import requests
import os
from werkzeug.utils import secure_filename

# Create a Blueprint for the dog classifier application
dog_classifier_app = Blueprint(
    "dog_classifier", __name__, template_folder="templates", static_folder="static"
)

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "static/uploads"
)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@dog_classifier_app.route("/")
def index():
    return render_template("dog_classifier.html")


@dog_classifier_app.route("/predict", methods=["POST"])
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
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Send the image to your deployed model
        with open(filepath, "rb") as img_file:
            files = {"file": (filename, img_file, "multipart/form-data")}
            response = requests.post("https://prashand.nl/predict", files=files)

        # Return the prediction results
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify(
                {"error": f"Model API returned error: {response.status_code}"}
            )

    return jsonify({"error": "Invalid file type. Allowed types: png, jpg, jpeg"})
