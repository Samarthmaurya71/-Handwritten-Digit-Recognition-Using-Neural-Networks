"""Train a CNN on the built-in MNIST handwritten digit dataset."""
import os
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "digit_model.keras")
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    print("Loading MNIST dataset (first run may download it)...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Scale pixel values and add the CNN channel dimension.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., None]
    x_test = x_test[..., None]

    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()
    model.fit(x_train, y_train, epochs=5, batch_size=128,
              validation_split=0.1, verbose=1)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    model.save(MODEL_PATH)

    print("\\nTraining complete.")
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print(f"Saved model: {MODEL_PATH}")

if __name__ == "__main__":
    main()
