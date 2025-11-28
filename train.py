import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K

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
img_size = (224, 224)
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
# Step 4: Build Model with Transfer Learning
# -----------------------------
def build_model(num_classes=3):
    try:
        # Try to load pre-trained MobileNetV2
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3),
            pooling=None
        )
        print("✅ Loaded pre-trained MobileNetV2 with ImageNet weights")
    except Exception as e:
        print(f"⚠️ Could not load pre-trained weights: {str(e)}")
        print("⚠️ Initializing with random weights (training from scratch)")
        base_model = MobileNetV2(
            include_top=False,
            weights=None,  # Random initialization
            input_shape=(224, 224, 3),
            pooling=None
        )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom head
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

model = build_model()

# -----------------------------
# Step 5: Compile Model
# -----------------------------
# First stage: Train only the head
print("\n🚀 Training head layers...")
model.compile(
    optimizer=Adam(learning_rate=1e-3),  # Fixed learning rate for initial training
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# -----------------------------
# Step 6: Train Model
# -----------------------------
def get_callbacks():
    return [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.keras',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        )
    ]

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=get_callbacks(),
    verbose=1
)

# Second stage: Fine-tune the top layers of the base model
print("\n🔧 Fine-tuning top layers...")
model.trainable = True
for layer in model.layers[:len(model.layers)-4]:
    layer.trainable = False

print("\n🔧 Fine-tuning with lower learning rate...")
model.compile(
    optimizer=Adam(learning_rate=1e-4),  # Lower learning rate for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=get_callbacks(),
    verbose=1
)

# -----------------------------
# Step 7: Save Model (Keras v3 format)
# -----------------------------
def save_model(model):
    try:
        # Save the model with reduced size
        keras_model_path = "potato_disease_model.keras"
        tflite_model_path = 'potato_disease_model.tflite'
        
        # Save Keras model
        model.save(keras_model_path, save_format="keras_v3")
        
        # Convert to TFLite for deployment
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        
        # Get file sizes
        keras_size = os.path.getsize(keras_model_path) / (1024*1024)
        tflite_size = os.path.getsize(tflite_model_path) / (1024*1024)
        
        print("✅ Models saved successfully!")
        print(f"   - Keras model size: {keras_size:.2f} MB")
        print(f"   - TFLite model size: {tflite_size:.2f} MB")
        
        # Check if models are under 50MB
        if keras_size > 50 or tflite_size > 50:
            print("⚠️ Warning: Model size exceeds 50MB. Consider reducing model complexity.")
        
    except Exception as e:
        print(f"❌ Error saving models: {str(e)}")
        # Try to save just the Keras model as fallback
        try:
            model.save("fallback_model.keras")
            print("✅ Saved fallback Keras model (may be larger in size)")
        except Exception as e2:
            print(f"❌ Could not save any model format: {str(e2)}")

save_model(model)

# -----------------------------
# Step 8: Evaluate Model
# -----------------------------
def evaluate_model(model, val_data):
    print("\n📊 Model Evaluation:")
    
    # Reset the generator to ensure we're at the start
    val_data.reset()
    
    # Get predictions for the entire validation set
    y_pred = model.predict(val_data, verbose=1)
    
    # Get the true labels
    y_true = val_data.classes
    
    # Get the class labels
    class_indices = val_data.class_indices
    
    # Evaluate the model
    results = model.evaluate(val_data, verbose=0)
    
    print(f"\nValidation Accuracy: {results[1]*100:.2f}%")
    print(f"Validation Loss: {results[0]:.4f}")
    if len(results) > 2:
        print(f"Validation AUC: {results[2]:.4f}")
    
    # Convert predictions to class indices
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate and print class-wise metrics
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred_classes, 
                              target_names=list(class_indices.keys())))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_indices.keys(),
                yticklabels=class_indices.keys())
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

evaluate_model(model, val_data)

# -----------------------------
# Step 9: Plot Training History
# -----------------------------
def plot_training_history(history, fine_tune_history=None):
    # Combine histories if fine-tuning was done
    if fine_tune_history is not None:
        for key in history.history:
            history.history[key].extend(fine_tune_history.history[key])
    
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

plot_training_history(history, history_fine if 'history_fine' in locals() else None)

# Print model summary
print("\n🧠 Model Architecture:")
model.summary()