from dog_classifier.app import dog_classifier_app
from flask import Flask, render_template, request, jsonify
import requests
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

# Create upload folder if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

app.register_blueprint(dog_classifier_app, url_prefix="/dog-classifier")


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
