import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ 1️⃣ Print TensorFlow Version
print(f"TensorFlow Version: {tf.__version__}")

# 🏋️‍♂️ 2️⃣ Data Augmentation (Reduced Distortion)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Reduced rotation
    width_shift_range=0.1,  # Reduced shift
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],  # Less variation
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)  # No augmentation for validation

# 📂 3️⃣ Load Dataset
train_dir = "train_test/train"
valid_dir = "train_test/validation"

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # No shuffle for evaluation
)

# ✅ **🔹 Print Class Indices to Verify Labels**
print("\n✅ Class Indices Mapping:")
print(train_generator.class_indices)

# ✅ **🔹 Get Actual Number of Classes**
num_classes = len(train_generator.class_indices)

# 🏆 4️⃣ Transfer Learning - MobileNetV2 as Feature Extractor
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze pretrained layers

# ✅ **🔹 Unfreeze Last 10 Layers for Fine-Tuning**
for layer in base_model.layers[-10:]:
    layer.trainable = True

# 🏗 5️⃣ Custom Classification Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)  # Fully connected layer
x = Dropout(0.5)(x)  # Dropout for regularization

# ✅ **🔹 Dynamically Set Output Layer**
predictions = Dense(num_classes, activation='softmax')(x)  # Adjusted for actual classes

# 🔗 6️⃣ Build Model
model = Model(inputs=base_model.input, outputs=predictions)

# ✅ **🔹 Reduce Learning Rate for Fine-Tuning**
model.compile(
    optimizer=Adam(learning_rate=0.00005),  # Lowered for stability
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 🎯 7️⃣ Train Model
epochs = 50  # ✅ **Increased to 50 for better learning**
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=valid_generator
)

# ✅ **🔹 Debug Predictions After Training**
print("\n🔍 Checking Sample Prediction...")
sample_img, _ = next(valid_generator)  # Get first batch
sample_prediction = model.predict(sample_img[:1])  # Predict on one image
predicted_class = np.argmax(sample_prediction)

print(f"Predicted Class Index: {predicted_class}")
print(f"Class Labels: {list(train_generator.class_indices.keys())}")

# 📊 8️⃣ Save Visualization: Training Accuracy and Loss
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Create 'static' folder if not exists
os.makedirs("athlesense/static", exist_ok=True)
plt.savefig("athlesense/static/accuracy_loss.png")  # Save figure
plt.close()  # Close plot

# 🔍 9️⃣ Evaluate Model
print("\n🔍 Evaluating Model on Validation Set...")
y_true = valid_generator.classes
y_pred = model.predict(valid_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification Report
class_labels = list(valid_generator.class_indices.keys())
print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

# 📉 1️⃣0️⃣ Save Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

plt.savefig("athlesense/static/confusion_matrix.png")  # Save figure
plt.close()  # Close plot

# 💾 1️⃣1️⃣ Save the Model
model.save("athlesense/sports_equipment_model.h5")

print("\n✅ Model training complete & saved as sports_equipment_model.h5")
print("📊 Training & evaluation visualizations saved in 'static/' folder.")
