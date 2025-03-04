from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import json
from test_image import classify_image
import shutil
import time

app = Flask(__name__)

# Ensure 'static' folder exists
os.makedirs("athlesense/static", exist_ok=True)

# 🔄 Function to Copy Visualizations from train_model.py to Static Folder
def load_training_visualizations():
    vis_files = ["accuracy_loss.png", "confusion_matrix.png"]
    for file in vis_files:
        src = file  # These should be saved by train_model.py
        dest = os.path.join("athlesense/static", file)
        if os.path.exists(src):
            shutil.copy(src, dest)

# 🏠 Index Route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle file upload
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            # Generate a unique filename (to avoid overwriting issues)
            timestamp = int(time.time())  # Unique timestamp
            filename = f"uploaded_{timestamp}.jpg"
            filepath = os.path.join("athlesense/static", filename)
            
            file.save(filepath)  # Save the uploaded file

            # ✅ Reset previous results before running classification
            result_file = os.path.join("athlesense/static", "results.json")
            with open(result_file, "w") as f:
                json.dump({}, f)  # Reset results

            # ✅ Run classification on the new image
            classify_image(filepath)

            # ✅ Redirect to results page with the correct image
            return redirect(url_for("result", img=filename))

    # Load visualizations from training process
    load_training_visualizations()

    return render_template("index.html")

# 🎯 Result Route
@app.route("/result")
def result():
    # Load classification results from JSON
    result_file = os.path.join("athlesense/static", "results.json")
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            results = json.load(f)
    else:
        results = {"predicted_class": "Unknown", "confidence_score": 0, "top_3": []}

    # ✅ Get latest uploaded image (passed in URL)
    img_filename = request.args.get("img", "uploaded_image.jpg")

    return render_template("result.html", results=results, image_file=img_filename)

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
