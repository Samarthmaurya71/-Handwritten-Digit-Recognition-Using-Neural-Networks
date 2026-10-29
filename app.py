
import os
import uuid
import numpy as np
import keras

from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image, ImageOps

# Project ke main folder ka path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixed model file ka path
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "digit_model_fixed.keras"
)

# Uploaded images save karne ka folder
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

# Allowed image formats
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

# Flask application
app = Flask(__name__)
app.secret_key = "change-this-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Model load karna
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
        print("Digit recognition model loaded successfully!")
    except Exception as exc:
        print(f"Could not load model: {exc}")
else:
    print(f"Model file not found: {MODEL_PATH}")


def allowed_file(filename):
    """Check whether the uploaded file has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def prepare_image(path):
    """Convert an uploaded digit image into 28x28 MNIST format."""

    image = Image.open(path).convert("L")
    image.thumbnail((20, 20))

    # Black background par digit ko center karna
    canvas = Image.new("L", (28, 28), color=0)

    # Dark digit on a light background ko invert karna
    pixels = np.asarray(image)

    if pixels.mean() > 127:
        image = ImageOps.invert(image)

    image = ImageOps.autocontrast(image)
    image.thumbnail((20, 20))

    x = (28 - image.width) // 2
    y = (28 - image.height) // 2

    canvas.paste(image, (x, y))

    # Pixel values ko 0 se 1 ke beech normalize karna
    arr = np.asarray(canvas).astype("float32") / 255.0

    # Model input shape: (1, 28, 28, 1)
    return arr.reshape(1, 28, 28, 1)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_url = None

    if request.method == "POST":

        if model is None:
            flash(
                "Model load nahi hua. models folder mein "
                "digit_model_fixed.keras check karein.",
                "error"
            )
            return redirect(url_for("index"))

        uploaded = request.files.get("image")

        if not uploaded or uploaded.filename == "":
            flash("Please choose a digit image first.", "error")
            return redirect(url_for("index"))

        if not allowed_file(uploaded.filename):
            flash(
                "Upload a PNG, JPG, JPEG, or BMP image.",
                "error"
            )
            return redirect(url_for("index"))

        extension = uploaded.filename.rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{extension}"
        path = os.path.join(UPLOAD_DIR, filename)

        uploaded.save(path)

        try:
            # Image se digit predict karna
            prediction = model.predict(
                prepare_image(path),
                verbose=0
            )[0]

            digit = int(np.argmax(prediction))
            confidence = float(prediction[digit]) * 100

            result = {
                "digit": digit,
                "confidence": round(confidence, 2),
                "scores": [
                    {
                        "digit": i,
                        "score": round(float(score) * 100, 2)
                    }
                    for i, score in enumerate(prediction)
                ]
            }

            image_url = url_for(
                "static",
                filename=f"uploads/{filename}"
            )

        except Exception as exc:
            flash(f"Could not process that image: {exc}", "error")

    return render_template(
        "index.html",
        result=result,
        image_url=image_url,
        model_ready=model is not None
    )


@app.errorhandler(413)
def too_large(_error):
    flash(
        "Image is too large. Maximum upload size is 8 MB.",
        "error"
    )
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)