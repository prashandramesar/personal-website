# dog_classifier/app.py
from flask import Blueprint, render_template
import os

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
