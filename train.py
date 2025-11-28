import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Step 1: Resize Images (Optional)
# -----------------------------
def resize_images(input_dir, output_dir, target_size=(128, 128)):
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        save_path = os.path.join(output_dir, class_name)
        os.makedirs(save_path, exist_ok=True)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                resized = cv2.resize(img, target_size)
                cv2.imwrite(os.path.join(save_path, img_name), resized)

# Uncomment to run resizing once
# resize_images("data/train", "data_resized/train")
# resize_images("data/test", "data_resized/test")

# -----------------------------
# Step 2: Define Paths
# -----------------------------
train_dir = "data_resized/train"
val_dir = "data_resized/val"

# -----------------------------
# Step 3: Preprocess Images
# -----------------------------
img_size = (128, 128)
batch_size = 32

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)
val_data = val_gen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)

# -----------------------------
# Step 4: Build CNN Model
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes
])

# -----------------------------
# Step 5: Compile Model
# -----------------------------
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -----------------------------
# Step 6: Train Model
# -----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# Step 7: Save Model (Keras v3 format)
# -----------------------------
model.save("potato_disease_model.keras", save_format="keras_v3")
print("✅ Model saved as potato_disease_model.keras (Keras v3 format)")

# -----------------------------
# Step 8: Evaluate Model
# -----------------------------
val_loss, val_acc = model.evaluate(val_data)
print(f"📊 Validation Accuracy: {val_acc*100:.2f}% | Loss: {val_loss:.4f}")

# -----------------------------
# Step 9: Plot Accuracy
# -----------------------------
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_plot.png")
plt.show()