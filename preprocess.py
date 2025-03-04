import os
import cv2
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm
import shutil

# Paths
DATASET_PATH = "dataset"  # Keep this folder unchanged
PROCESSED_PATH = "processed_data"
REMOVED_IMAGES_FILE = "removed_images.txt"
IMAGE_SIZE = (224, 224)

# List to store hashes of known logos (to remove similar images)
logo_hashes = []

# Function to compute perceptual hash of an image
def get_image_hash(image_path):
    try:
        img = Image.open(image_path).convert("L").resize((8, 8))  # Convert to grayscale, resize for hash
        return imagehash.phash(img)  # Perceptual hash
    except Exception as e:
        print(f"❌ Error computing hash for {image_path}: {e}")
        return None

# Function to preprocess dataset images
def preprocess_images():
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset path '{DATASET_PATH}' not found!")
        return
    
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    removed_images = []  # Track removed images

    categories = os.listdir(DATASET_PATH)
    
    for category in tqdm(categories, desc="Processing Categories"):
        input_folder = os.path.join(DATASET_PATH, category)
        output_folder = os.path.join(PROCESSED_PATH, category)
        os.makedirs(output_folder, exist_ok=True)

        for img_name in os.listdir(input_folder):
            img_path = os.path.join(input_folder, img_name)
            processed_img_path = os.path.join(output_folder, img_name)

            # Copy the image first (so original dataset remains unchanged)
            shutil.copy2(img_path, processed_img_path)

            # Try to open and verify the copied image
            try:
                img = Image.open(processed_img_path)
                img.verify()  # Check if corrupted
                img = Image.open(processed_img_path).convert("RGB")  # Convert to RGB
            except (IOError, SyntaxError):
                print(f"❌ Corrupt image removed: {processed_img_path}")
                removed_images.append(img_path)
                os.remove(processed_img_path)
                continue  # Skip corrupt images
            
            # Convert to numpy array for OpenCV processing
            img = np.array(img)
            
            # Remove grayscale images
            if len(img.shape) == 2 or img.shape[2] == 1:
                print(f"⚠ Grayscale image removed: {processed_img_path}")
                removed_images.append(img_path)
                os.remove(processed_img_path)
                continue  # Skip grayscale images
            
            # Remove small images
            if img.shape[0] < 100 or img.shape[1] < 100:
                print(f"⚠ Small image removed: {processed_img_path}")
                removed_images.append(img_path)
                os.remove(processed_img_path)
                continue  # Skip small images
            
            # Remove logo-like images using perceptual hash
            img_hash = get_image_hash(processed_img_path)
            if img_hash and img_hash in logo_hashes:
                print(f"⚠ Logo-like image removed: {processed_img_path}")
                removed_images.append(img_path)
                os.remove(processed_img_path)
                continue  # Skip logo images

            # Resize image to (224, 224)
            img_resized = cv2.resize(img, IMAGE_SIZE)

            # Normalize pixel values (0-1)
            img_resized = img_resized.astype(np.float32) / 255.0  

            # Save the processed image
            try:
                success = cv2.imwrite(processed_img_path, (img_resized * 255).astype(np.uint8))  # Convert back to (0-255) scale
                if success:
                    print(f"📸 Saved: {processed_img_path}")
                else:
                    print(f"❌ Failed to save: {processed_img_path}")
            except Exception as e:
                print(f"❌ Error saving image {processed_img_path}: {e}")

        print(f"✅ Processed category: {category}")

    # Save log of removed images
    if removed_images:
        with open(REMOVED_IMAGES_FILE, "w") as file:
            for img in removed_images:
                file.write(img + "\n")
        print(f"\n🗑 Removed/corrupt images logged in '{REMOVED_IMAGES_FILE}'.")

# Run preprocessing
print("\n🔄 Starting dataset preprocessing...")
preprocess_images()
print("\n🎯 Preprocessing complete! Processed images are saved in 'processed_data/'.")
