import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
from whatsapp_bot import predict_image  # Reuse the prediction logic

def load_model(model_path='potato_disease_model.keras'):
    """Load the trained model with error handling."""
    try:
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def test_prediction(image_path, model_path='potato_disease_model.keras'):
    """
    Test prediction on a single image.
    
    Args:
        image_path (str): Path to the test image
        model_path (str): Path to the model file
    """
    # Verify image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
        
    print(f"\nTesting image: {image_path}")
    
    # Load image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # Make prediction
    predicted_class, confidence = predict_image(image_bytes)
    
    # Display results
    if predicted_class:
        print(f"\nPredicted: {predicted_class}")
        print("Confidence levels:")
        for cls, conf in confidence.items():
            print(f"- {cls}: {conf:.2f}%")
    else:
        print("❌ Failed to make prediction")

def test_with_sample_images(test_dir='test_images'):
    """
    Test the model with multiple images from a directory.
    Directory structure should be:
    test_images/
        Early_Blight/
        Late_Blight/
        Healthy/
    """
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return
        
    print(f"\nTesting with images from: {test_dir}")
    
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        print(f"\n--- Testing {class_name} ---")
        for img_name in os.listdir(class_dir)[:3]:  # Test first 3 images per class
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                test_prediction(img_path)

if __name__ == "__main__":
    # Test with a single image
    test_image = "test_image.jpg"  # Change this to your test image path
    if os.path.exists(test_image):
        test_prediction(test_image)
    else:
        print(f"Test image not found: {test_image}")
        print("Please provide a valid image path or use the test_with_sample_images() function")
    
    # Uncomment to test with multiple images from a directory
    # test_with_sample_images('test_images')