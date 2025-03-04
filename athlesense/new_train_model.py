import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 🏋️‍♂️ 1️⃣ Detect New Categories
train_dir = "train_test/train"
valid_dir = "train_test/validation"

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

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
    shuffle=False
)

# Get the number of classes dynamically
num_classes = len(train_generator.class_indices)

# 🏆 2️⃣ Load Existing Model
existing_model_path = "athlesense/sports_equipment_model.h5"
model = tf.keras.models.load_model(existing_model_path)

# Modify the last layer to accommodate new categories
x = model.layers[-2].output  # Get the second last layer
new_predictions = Dense(num_classes, activation='softmax')(x)

# Create updated model
updated_model = Model(inputs=model.input, outputs=new_predictions)

# Unfreeze some layers for fine-tuning
for layer in updated_model.layers[:-5]:  
    layer.trainable = False  # Keep most layers frozen

# ⚡ 3️⃣ Compile Updated Model
updated_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 🎯 4️⃣ Fine-Tune Model
epochs = 5  # Fine-tune for fewer epochs to save time
history = updated_model.fit(
    train_generator,
    epochs=epochs,
    validation_data=valid_generator
)

# Create 'static' folder if not exists
os.makedirs("athlesense/static", exist_ok=True)

# 📊 5️⃣ Save Visualization: Accuracy & Loss
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Updated Model: Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Updated Model: Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig("athlesense/static/updated_accuracy_loss.png")
plt.close()

# 🔍 6️⃣ Evaluate Updated Model
y_true = valid_generator.classes
y_pred = updated_model.predict(valid_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification Report
class_labels = list(valid_generator.class_indices.keys())
print("\n📊 Updated Classification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

# 📉 7️⃣ Save Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title("Updated Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

plt.savefig("athlesense/static/updated_confusion_matrix.png")
plt.close()

# 💾 8️⃣ Save Updated Model
updated_model.save("athlesense/sports_equipment_model_updated.h5")

print("\n✅ Model updated & saved as sports_equipment_model_updated.h5")
print("📊 Updated visualizations saved in 'static/' folder.")
