import os
import json
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("athlesense/sports_equipment_model.h5")

# Define class labels
class_labels = [
    "Badminton_Racket", "Baseball_ball", "Baseball_Bat", "Basketball_ball",
    "Billiard_Cue", "Bow_and_Arrow_Archery", "Boxing_Gloves", "Carrom_Board",
    "Carrom_Coins", "Chess_Board", "Cricket_Ball", "Cricket_Bat", 
    "Hockey_Ball", "Hockey_Stick", "Shuttlecock", "Skateboard", 
    "Soccer_Ball", "Squash_Racket", "Table_Tennis_Ball", "Table_Tennis_Paddle", 
    "Tennis_Ball", "Tennis_Racket",  "Volleyball_ball"      
]

# Function to preprocess image for model
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for proper display
    img_resized = cv2.resize(img, (224, 224))  # Resize to match model input size
    img_normalized = img_resized / 255.0  # Normalize pixel values
    img_expanded = np.expand_dims(img_normalized, axis=0)  # Add batch dimension
    return img, img_expanded  # Return original and processed image

# 📂 Create 'static' folder if it doesn't exist
os.makedirs("static", exist_ok=True)

# Function to classify an image
def classify_image(image_path):
    # Preprocess image
    original_image, processed_image = preprocess_image(image_path)

    # Save the uploaded image for display in Flask
    cv2.imwrite("athlesense/static/uploaded_image.jpg", cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))

    # Predict
    predictions = model.predict(processed_image)[0]  # Extract the single prediction
    predicted_class_idx = np.argmax(predictions)  # Get highest probability index
    confidence_score = predictions[predicted_class_idx] * 100  # Convert to percentage

    # Get Top-3 predictions
    top_3_indices = np.argsort(predictions)[-3:][::-1]  # Sort and get top 3
    top_3_classes = [class_labels[i] for i in top_3_indices]
    top_3_confidences = [predictions[i] * 100 for i in top_3_indices]  # Convert to %

    # Save class-wise probability bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(class_labels, predictions * 100, color='skyblue')
    plt.xlabel("Classes")
    plt.ylabel("Probability (%)")
    plt.xticks(rotation=45)
    plt.title("Class-Wise Probabilities")
    plt.savefig("static/class_probabilities.png")  # Save the plot
    plt.close()

    # Save top-3 predictions bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(top_3_classes, top_3_confidences, color=['green', 'orange', 'red'])
    plt.xlabel("Top-3 Predicted Classes")
    plt.ylabel("Confidence Score (%)")
    plt.title("Top-3 Predictions with Confidence Scores")
    plt.ylim(0, 100)
    plt.savefig("athlesense/static/top3_predictions.png")  # Save the plot
    plt.close()

    # Save results as JSON for Flask to display
    results = {
        "predicted_class": class_labels[predicted_class_idx],
        "confidence_score": round(confidence_score, 2),
        "top_3": [{"class": top_3_classes[i], "confidence": round(top_3_confidences[i], 2)} for i in range(3)]
    }

    with open("athlesense/static/results.json", "w") as f:
        json.dump(results, f)

    print("\n✅ Image classified. Results saved in 'athlesense/static/' folder.")

    return results


