import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, 
                                   Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                      ModelCheckpoint)
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# -----------------------------
# Step 1: Resize Images (Optional)
# -----------------------------
def resize_images(input_dir, output_dir, target_size=(128, 128)):
    """Resize images to target size and save to output directory."""
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        save_path = os.path.join(output_dir, class_name)
        os.makedirs(save_path, exist_ok=True)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    resized = cv2.resize(img, target_size)
                    # Convert to RGB (if needed)
                    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join(save_path, img_name), resized)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

# Uncomment to run resizing once
# resize_images("data/train", "data_resized/train")
# resize_images("data/val", "data_resized/val")

# -----------------------------
# Step 2: Define Paths and Parameters
# -----------------------------
train_dir = "data_resized/train"
val_dir = "data_resized/val"
img_size = (128, 128)
batch_size = 32
num_classes = 3

# -----------------------------
# Step 3: Data Augmentation and Preprocessing
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# -----------------------------
# Step 4: Enhanced Model Architecture
# -----------------------------
def create_model(input_shape=(128, 128, 3), num_classes=3):
    model = Sequential([
        # First Conv Block
        Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second Conv Block
        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),
        
        # Third Conv Block
        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),
        
        # Dense Layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

model = create_model()

# -----------------------------
# Step 5: Callbacks
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# -----------------------------
# Step 6: Compile Model
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# -----------------------------
# Step 7: Train Model
# -----------------------------
print("🚀 Starting model training...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# -----------------------------
# Step 8: Save Final Model
# -----------------------------
model.save("potato_disease_model.keras", save_format="keras_v3")
print("✅ Model saved as potato_disease_model.keras")

# -----------------------------
# Step 9: Evaluate Model
# -----------------------------
print("\n📊 Model Evaluation:")
val_loss, val_acc, val_precision, val_recall = model.evaluate(val_data)
print(f"Validation Accuracy: {val_acc*100:.2f}%")
print(f"Validation Precision: {val_precision*100:.2f}%")
print(f"Validation Recall: {val_recall*100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

# -----------------------------
# Step 10: Generate Classification Report and Confusion Matrix
# -----------------------------
def evaluate_model(model, val_data):
    # Get predictions
    y_pred = model.predict(val_data)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_data.classes
    
    # Classification Report
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred_classes, 
                              target_names=list(val_data.class_indices.keys())))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=val_data.class_indices.keys(),
                yticklabels=val_data.class_indices.keys())
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()

evaluate_model(model, val_data)

# -----------------------------
# Step 11: Plot Training History
# -----------------------------
def plot_training_history(history):
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

plot_training_history(history)