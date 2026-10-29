
import json
import zipfile
from pathlib import Path

import keras

# Original trained model ko safe rakho.
model_path = Path("models/digit_model.keras")
fixed_path = Path("models/digit_model_fixed.keras")

if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

def clean_config(obj):
    """Remove initializer settings unsupported by this Keras version."""
    if isinstance(obj, dict):
        obj.pop("input_axes", None)
        obj.pop("output_axes", None)

        for value in obj.values():
            clean_config(value)

    elif isinstance(obj, list):
        for value in obj:
            clean_config(value)

# .keras file ek ZIP archive hoti hai.
with zipfile.ZipFile(model_path, "r") as source:
    with zipfile.ZipFile(
        fixed_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as destination:

        for item in source.infolist():
            data = source.read(item.filename)

            if item.filename == "config.json":
                config = json.loads(data.decode("utf-8"))
                clean_config(config)
                data = json.dumps(config).encode("utf-8")

            destination.writestr(item, data)

print("Updated model configuration saved.")
print("Testing the updated model...")

model = keras.models.load_model(
    str(fixed_path),
    compile=False
)

print("SUCCESS: Model loaded correctly!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

# Check that the model can make a prediction.
import numpy as np

sample = np.zeros((1, 28, 28, 1), dtype=np.float32)
prediction = model.predict(sample, verbose=0)

print("Prediction test passed.")
print("Prediction output shape:", prediction.shape)
print("Fixed model:", fixed_path)
