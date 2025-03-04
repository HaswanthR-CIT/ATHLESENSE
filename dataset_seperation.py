import os
import shutil
import random

# Paths
DATASET_PATH = "processed_data"  # Use the cleaned dataset
TRAIN_PATH = "train_test/train"
VAL_PATH = "train_test/validation"

# Define class labels
classes = [
    "Badminton_Racket", "Baseball_ball", "Baseball_Bat", "Basketball_ball",
    "Billiard_Cue", "Bow_and_Arrow_Archery", "Boxing_Gloves", "Carrom_Board",
    "Carrom_Coins", "Chess_Board", "Cricket_Ball", "Cricket_Bat", 
    "Hockey_Ball", "Hockey_Stick", "Shuttlecock", "Skateboard", 
    "Soccer_Ball", "Squash_Racket", "Table_Tennis_Ball", "Table_Tennis_Paddle", 
    "Tennis_Ball", "Tennis_Racket",  "Volleyball_ball"      
]


# Train-validation split ratio
SPLIT_RATIO = 0.8  # 80% training, 20% validation

def split_dataset():
    for category in classes:
        category_path = os.path.join(DATASET_PATH, category)
        if not os.path.exists(category_path):
            print(f"⚠ Skipping missing category: {category}")
            continue

        # Collect all images
        images = os.listdir(category_path)
        random.shuffle(images)

        # Split into train and validation
        train_size = int(len(images) * SPLIT_RATIO)
        train_images = images[:train_size]
        val_images = images[train_size:]

        # Create train & validation folders
        os.makedirs(os.path.join(TRAIN_PATH, category), exist_ok=True)
        os.makedirs(os.path.join(VAL_PATH, category), exist_ok=True)

        # Copy images instead of moving them
        for img in train_images:
            shutil.copy2(os.path.join(category_path, img), os.path.join(TRAIN_PATH, category, img))

        for img in val_images:
            shutil.copy2(os.path.join(category_path, img), os.path.join(VAL_PATH, category, img))

        print(f"✅ {category}: {len(train_images)} train, {len(val_images)} validation images copied")

# Run the dataset split
print("\n🔄 Splitting dataset into training & validation sets...")
split_dataset()
print("\n🎯 Dataset successfully split! (Original images remain in processed_data)")
