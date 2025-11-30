import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Set random seed for reproducibility
tf.random.set_seed(42)

# Constants
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 25

# Data paths - adjust these paths to where your data is located
data_dir = 'data'  # This should contain 'train' and 'val' subdirectories
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# Class names must match the subdirectory names exactly
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Print directory structure for debugging
print("Current working directory:", os.getcwd())
print("Train directory:", os.path.abspath(train_dir))
print("Validation directory:", os.path.abspath(val_dir))

# Check if directories exist and contain data
def check_directory(directory):
    if not os.path.exists(directory):
        print(f" Directory does not exist: {directory}")
        return False
    
    classes = [d for d in os.listdir(directory) 
              if os.path.isdir(os.path.join(directory, d))]
    
    if not classes:
        print(f" No class directories found in: {directory}")
        return False
    
    print(f" Found {len(classes)} classes in {directory}:")
    for cls in classes:
        cls_path = os.path.join(directory, cls)
        num_files = len([f for f in os.listdir(cls_path) 
                        if os.path.isfile(os.path.join(cls_path, f))])
        print(f"   - {cls}: {num_files} images")
    
    return True

# Check data directories
if not (check_directory(train_dir) and check_directory(val_dir)):
    print(" Please check your data directories and try again.")
    exit(1)

# Enhanced data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2  # Use 20% of training data for validation
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

# Load training data
print("\nLoading training data...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=True,
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False,
    subset='validation'
)

# Calculate class weights with more emphasis on minority classes
class_weights = {}
total_samples = sum([len(files) for _, _, files in os.walk(train_dir) if files])
for i, cls in enumerate(CLASS_NAMES):
    class_path = os.path.join(train_dir, cls)
    if os.path.exists(class_path):
        num_samples = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
        class_weights[i] = (1 / num_samples) * (total_samples / len(CLASS_NAMES))

print("Class weights:", class_weights)

# Reset generators
train_generator.reset()
validation_generator.reset()

# Build transfer learning model (EfficientNetB0 with fine-tuning)
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(256, 256, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'  # Use global average pooling directly
)

# Unfreeze some top layers for fine-tuning
base_model.trainable = False
for layer in base_model.layers[-20:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

model = Sequential([
    base_model,
    Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# Compile model with class weights
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# Learning rate scheduler
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# Early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Model checkpoint
model_checkpoint = ModelCheckpoint(
    'potato_disease_model.keras',
    save_best_only=True,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

callbacks = [reduce_lr, early_stopping, model_checkpoint]

# Print model summary
model.summary()

# Train model with class weights
print("\nStarting model training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    class_weight=class_weights
)

# Save the final model
model.save('potato_disease_model_final.keras')
print("✅ Training completed and model saved!")

# Evaluate the model
print("\nEvaluating model...")
validation_generator.reset()
val_preds = model.predict(validation_generator, verbose=1)
y_pred = np.argmax(val_preds, axis=1)
y_true = validation_generator.classes
class_names = list(validation_generator.class_indices.keys())

# Print evaluation metrics
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Generate and save confusion matrix plot
try:
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save the plot
    plt.savefig('confusion_matrix.png')
    print("✅ Confusion matrix saved as 'confusion_matrix.png'")
    plt.close()
    
except Exception as e:
    print(f"⚠️ Could not generate confusion matrix: {e}")

# Plot training history
try:
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("✅ Training history plot saved as 'training_history.png'")
    plt.close()
    
except Exception as e:
    print(f"⚠️ Could not generate training plots: {e}")