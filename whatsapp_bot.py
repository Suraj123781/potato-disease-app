# whatsapp_bot.py
import os
import io
import numpy as np
import requests
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# Disease information
DISEASE_INFO = {
    "Early Blight": {
        "name": "Early Blight",
        "description": "Early blight is a common fungal disease that affects potato plants, causing dark spots with concentric rings on leaves.",
        "prevention": [
            "Rotate crops every 2-3 years",
            "Remove and destroy infected plant debris",
            "Water at the base of plants to keep foliage dry",
            "Apply fungicides at the first sign of disease"
        ]
    },
    "Late Blight": {
        "name": "Late Blight",
        "description": "Late blight is a serious fungal disease that can destroy entire potato crops, causing water-soaked spots that turn brown and mushy.",
        "prevention": [
            "Plant certified disease-free seed potatoes",
            "Space plants for good air circulation",
            "Apply fungicides preventatively in wet weather",
            "Remove and destroy infected plants immediately"
        ]
    },
    "Healthy": {
        "name": "Healthy Potato Plant",
        "description": "Your potato plant appears to be healthy with no signs of disease.",
        "prevention": [
            "Continue good cultural practices",
            "Monitor regularly for signs of disease",
            "Water consistently but avoid overwatering",
            "Maintain proper plant spacing"
        ]
    }
}

# Configure environment for CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations for consistent CPU results

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Class names for predictions (must match training)
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

# Initialize model
print("🔍 Loading custom potato disease model...")
try:
    model = load_model('potato_disease_model.keras')
    print("✅ Custom model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("⚠️ Make sure you've trained the model first by running train.py")
    exit(1)

def predict_disease(image_bytes):
    try:
        # Load and preprocess the image (match training preprocessing)
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((256, 256))  # Match training image size
        img_array = np.array(img) / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        
        # Map model outputs to class names (must match training order)
        class_mapping = [
            "Early Blight",  # Index 0
            "Late Blight",   # Index 1
            "Healthy"        # Index 2
        ]
        
        # Get probabilities and predicted class
        probabilities = predictions[0]
        predicted_class_idx = int(np.argmax(probabilities))
        predicted_class = class_mapping[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx]) * 100
        
        # Create results dictionary with all class probabilities
        results = {class_name: float(prob) * 100 
                 for class_name, prob in zip(class_mapping, probabilities)}
        
        # Debug output
        print("\n--- Prediction Debug Info ---")
        print(f"Raw predictions: {probabilities}")
        print(f"Predicted class index: {predicted_class_idx}")
        print(f"Mapped class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        print("Class probabilities:", results)
        
        print(f"Raw predictions: {predictions[0]}")
        print(f"Predicted class index: {predicted_class_idx}")
        print(f"Mapped class: {predicted_class}")
        
        print(f"✅ Prediction: {predicted_class} | {results}")
        return predicted_class, results
        
    except Exception as e:
        print("❌ Error processing image:", e)
        # Return mock data in case of error
        mock_results = {
            "Early Blight": 33.3,
            "Late Blight": 33.3,
            "Healthy": 33.3
        }
        return "Error processing image", mock_results

@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    """Handle incoming WhatsApp messages"""
    try:
        # Get request data
        sender = request.values.get("From", "")
        incoming_msg = request.values.get("Body", "").strip().lower()
        num_media = int(request.values.get("NumMedia", 0))
        
        print(f"📨 From: {sender}")
        print(f"💬 Message: {incoming_msg}")
        print(f"📷 Media count: {num_media}")

        resp = MessagingResponse()
        msg = resp.message()

        # Handle image upload
        if num_media > 0:
            media_url = request.values.get("MediaUrl0")
            print(f"📥 Downloading image: {media_url}")
            
            try:
                # Download image
                response = requests.get(media_url)
                if response.status_code != 200:
                    raise Exception(f"Failed to download image: {response.status_code}")
                
                # Predict disease
                predicted_class, confidence_scores = predict_disease(response.content)
                
                if predicted_class and confidence_scores:
                    # Get confidence for the predicted class
                    confidence = confidence_scores.get(predicted_class, 0)
                    
                    # Get disease info
                    disease_info = DISEASE_INFO.get(predicted_class, DISEASE_INFO["Healthy"])
                    
                    # Format response
                    response_msg = (
                        f"🌱 *{disease_info['name']}* ({confidence:.1f}% confidence)\n\n"
                        f"📝 {disease_info['description']}\n\n"
                        "🔍 *Prevention Tips:*\n"
                    )
                    
                    # Add prevention tips
                    for i, tip in enumerate(disease_info['prevention'], 1):
                        response_msg += f"{i}. {tip}\n"
                    
                    msg.body(response_msg)
                    print(f"✅ Sent prediction: {predicted_class} ({confidence:.1f}%)")
                else:
                    msg.body("❌ Could not process the image. Please try with a clearer photo of a potato leaf.")
                    
            except Exception as e:
                print(f"❌ Error processing image: {e}")
                msg.body("⚠️ An error occurred while processing your image. Please try again.")
            
            return str(resp)

        # Handle text commands
        if not incoming_msg or "hi" in incoming_msg or "hello" in incoming_msg or "help" in incoming_msg:
            help_text = """🌱 *Potato Disease Detection Bot*\n\n"""
            help_text += "To use this bot:\n"
            help_text += "1. Send a clear photo of a potato leaf\n"
            help_text += "2. The bot will analyze it and provide disease information\n"
            help_text += "3. You'll receive prevention tips based on the diagnosis\n\n"
            help_text += "Note: This is for educational purposes only. For severe cases, consult an agricultural expert."
            msg.body(help_text)
            
        else:
            msg.body("🤖 I can help you identify potato plant diseases. Please send a photo of a potato leaf for analysis.")
            
        return str(resp)

    except Exception as e:
        print(f"❌ Error in webhook: {e}")
        resp = MessagingResponse()
        resp.message("⚠️ An unexpected error occurred. Please try again later.")
        return str(resp)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)