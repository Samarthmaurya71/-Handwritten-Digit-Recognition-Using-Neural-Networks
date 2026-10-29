# Handwritten Digit Recognition Using CNN

A complete beginner-friendly deep learning project with a light-theme web interface. Upload an image of a handwritten digit (0–9) and the trained CNN predicts the digit and confidence scores.

## Features
- CNN model trained on the built-in MNIST dataset
- Image upload (PNG, JPG, JPEG, BMP)
- Prediction from 0 to 9 with confidence percentage
- Confidence score chart for all 10 classes
- Light, responsive UI
- Input validation and upload size limit

## Requirements
- Windows 10/11
- Python 3.10–3.12 recommended for best TensorFlow compatibility
- Internet access on the first training run to download MNIST

> TensorFlow support can lag behind the newest Python versions. If your Python version is 3.13+, create the environment with Python 3.12 if available.

## Run on Windows (VS Code / CMD)

Open CMD in this project folder:

```bat
python -m venv venv
venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_model.py
python app.py
```

When training finishes, open the local address printed by Flask, usually:
`http://127.0.0.1:5000`

Training may take several minutes. The first run downloads MNIST automatically. The trained model is saved as `models/digit_model.keras`.

## How to test
1. Use a clear image containing one large handwritten digit.
2. Use a plain background and dark digit strokes.
3. Crop the image close to the digit, without extra text.
4. Upload the image on the website and click **Predict digit**.

The model was trained on 28×28 grayscale MNIST images, so photos, colored backgrounds, multiple digits, or unusual handwriting may reduce accuracy.

## Project structure

```text
Handwritten_Digit_Recognition_Light/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── models/
│   └── (digit_model.keras is created after training)
├── static/
│   ├── style.css
│   └── uploads/
│       └── .gitkeep
└── templates/
    └── index.html
```

## Workflow
1. Load the MNIST dataset.
2. Normalize pixels and reshape the images.
3. Train a Convolutional Neural Network.
4. Evaluate on the held-out test set.
5. Save the model.
6. Launch the Flask web application.
7. Preprocess the uploaded image and display prediction results.

## Troubleshooting
- **Model is not trained yet:** run `python train_model.py` first.
- **TensorFlow installation fails:** use Python 3.12 (64-bit), recreate the virtual environment, then reinstall requirements.
- **Image prediction is poor:** upload a centered, single digit on a plain background.
- **Port 5000 is busy:** change `app.run(debug=True)` to `app.run(debug=True, port=5001)` and open port 5001.

This is an educational demo, not a production-secured deployment. Before public hosting, disable debug mode, use a strong secret key, and add production server configuration.
