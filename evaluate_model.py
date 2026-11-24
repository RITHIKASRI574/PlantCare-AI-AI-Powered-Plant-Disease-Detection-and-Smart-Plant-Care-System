
from pathlib import Path
import tensorflow as tf
import numpy as np
import json

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "plant_disease_model.keras"
TEST_DIR = BASE_DIR / "dataset_split" / "test"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load category names
with open(BASE_DIR / "model" / "class_names.json", "r") as file:
    class_names = json.load(file)

# Load test images
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# Normalize test images
test_dataset = test_dataset.map(
    lambda images, labels: (images / 255.0, labels)
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_dataset)

print("\nTEST RESULTS")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")

# Get actual labels and predictions
actual_labels = np.concatenate([
    labels.numpy() for images, labels in test_dataset
])

predictions = model.predict(test_dataset)
predicted_labels = np.argmax(predictions, axis=1)

# Classification report
print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        actual_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )
)

# Confusion matrix
print("\nCONFUSION MATRIX")
print(confusion_matrix(actual_labels, predicted_labels))

print("\nMODEL EVALUATION COMPLETED!")
