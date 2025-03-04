import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# ✅ Load the trained model
MODEL_PATH = "athlesense/sports_equipment_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"🚨 Model file not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Load class labels (same order as training)
train_dir = "train_test/train"
class_labels = sorted(os.listdir(train_dir))  # Ensure labels match training order

print("\n✅ Class Labels:", class_labels)

# ✅ Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (224, 224))  # Resize to model's input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# ✅ Function to predict and display the image
def predict_image(image_path):
    img = preprocess_image(image_path)
    
    # Model prediction
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)  # Get index of highest probability
    predicted_class = class_labels[predicted_class_index]  # Map to class label
    confidence = np.max(predictions) * 100  # Confidence %

    # Display image with prediction
    plt.imshow(cv2.imread(image_path)[:, :, ::-1])  # Convert BGR to RGB for display
    plt.axis("off")
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%")
    plt.show()

    print(f"\n✅ Predicted Class: {predicted_class} (Confidence: {confidence:.2f}%)")

# ✅ Test with an image
TEST_IMAGE_PATH = "image9.jpg"  # Change this to your test image path

if not os.path.exists(TEST_IMAGE_PATH):
    raise FileNotFoundError(f"🚨 Test image not found at {TEST_IMAGE_PATH}")

predict_image(TEST_IMAGE_PATH)
