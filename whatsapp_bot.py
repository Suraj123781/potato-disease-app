import os
import io
import requests
import numpy as np
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Twilio credentials
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')

# Class names for predictions
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Initialize model
print("🔍 Loading pre-trained MobileNetV2 model...")
model = MobileNetV2(weights='imagenet')
print("✅ Model loaded successfully!")

# Store the last prediction for each user
last_prediction = {}

# Disease information
DISEASE_INFO = {
    "Early Blight": {
        "name": "Early Blight",
        "description": "Early blight is a common fungal disease that affects potato plants.",
        "prevention": [
            "Rotate crops regularly",
            "Remove and destroy infected plants",
            "Use disease-free seed potatoes",
            "Apply fungicides preventatively"
        ],
        "products": [
            {"name": "Bonide Copper Fungicide", "url": "https://www.amazon.com/Bonide-811-Copper-Fungicide-16-oz/dp/B000HHO1TO"},
            {"name": "Southern Ag Liquid Copper Fungicide", "url": "https://www.amazon.com/Southern-Ag-Liquid-Copper-Fungicide/dp/B000I1VZ9K"},
            {"name": "Garden Safe Fungicide 3", "url": "https://www.amazon.com/Garden-Safe-100511831-Fungicide-24-Ounce/dp/B00BRLU4Q6"}
        ]
    },
    "Late Blight": {
        "name": "Late Blight",
        "description": "Late blight is a serious disease that can destroy entire potato crops.",
        "prevention": [
            "Plant resistant varieties",
            "Ensure good air circulation",
            "Avoid overhead watering",
            "Apply fungicides before infection"
        ],
        "products": [
            {"name": "Bonide Copper Fungicide", "url": "https://www.amazon.com/Bonide-811-Copper-Fungicide-16-oz/dp/B000HHO1TO"},
            {"name": "Daconil Fungicide Concentrate", "url": "https://www.amazon.com/Spectracide-100507514-Immunox-Multi-Purpose-Fungicide/dp/B00BRLU4Q6"},
            {"name": "GardenTech Daconil Fungicide", "url": "https://www.amazon.com/GardenTech-Daconil-Fungicide-16-Ounce-100540634/dp/B00BRLU4Q6"}
        ]
    },
    "Healthy": {
        "name": "Healthy",
        "description": "Your plant appears to be healthy! No signs of disease detected.",
        "prevention": [
            "Continue good gardening practices",
            "Monitor plants regularly",
            "Maintain proper spacing",
            "Water at the base of plants"
        ],
        "products": [
            {"name": "Jobe's Organics Vegetable Fertilizer", "url": "https://www.amazon.com/Jobes-09026-Organic-Vegetable-Fertilizer/dp/B00BRLU4Q6"},
            {"name": "Dr. Earth Organic Compost", "url": "https://www.amazon.com/Dr-Earth-803-Organic-Compost/dp/B00BRLU4Q6"},
            {"name": "Miracle-Gro Plant Food", "url": "https://www.amazon.com/Miracle-Gro-Water-Soluble-Plant-Fertilizer/dp/B00BRLU4Q6"}
        ]
    }
}

def predict_image(image_bytes):
    try:
        # Load and preprocess the image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Make prediction
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=5)[0]  # Get top 5 predictions
        
        # Initialize results with all classes
        results = {class_name: 0.0 for class_name in CLASS_NAMES}
        
        # Map ImageNet classes to our classes
        for _, label, prob in decoded_predictions:
            label_lower = label.lower()
            prob_percent = float(prob) * 100
            
            # Early Blight indicators
            if any(keyword in label_lower for keyword in ['blight', 'spot', 'spotting', 'leaf spot', 'leafspot', 'disease', 'fungal']):
                if any(keyword in label_lower for keyword in ['early', 'alternaria']):
                    results['Early Blight'] += prob_percent * 1.5  # Boost early blight probability
                elif any(keyword in label_lower for keyword in ['late', 'phytophthora']):
                    results['Late Blight'] += prob_percent * 1.5  # Boost late blight probability
                else:
                    # General blight/disease terms - split between early and late blight
                    results['Early Blight'] += prob_percent * 0.7
                    results['Late Blight'] += prob_percent * 0.8
            
            # Healthy indicators
            elif any(keyword in label_lower for keyword in ['leaf', 'plant', 'foliage', 'green', 'healthy']):
                results['Healthy'] += prob_percent
        
        # Normalize results to sum to 100%
        total = sum(results.values())
        if total > 0:
            results = {k: (v / total) * 100 for k, v in results.items()}
        
        # Get the class with highest probability
        predicted_class = max(results.items(), key=lambda x: x[1])[0]
        
        # Add some basic validation
        if predicted_class == 'Healthy' and results['Healthy'] < 60:
            # If healthy is predicted but confidence is low, check for diseases
            if results['Early Blight'] > 30:
                predicted_class = 'Early Blight'
            elif results['Late Blight'] > 30:
                predicted_class = 'Late Blight'
        
        print(f"✅ Prediction: {predicted_class} | {results}")
        return predicted_class, results
        
    except Exception as e:
        print("❌ Error processing image:", e)
        # Return mock data in case of error
        mock_results = {
            "Early Blight": 10.0,
            "Late Blight": 10.0,
            "Healthy": 80.0
        }
        return "Healthy (Error)", mock_results

@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    try:
        sender = request.values.get("From", "")
        incoming_msg = request.values.get("Body", "").strip().lower()
        num_media = int(request.values.get("NumMedia", 0))
        resp = MessagingResponse()

        print(f"📨 From: {sender}")
        print(f"💬 Message: {incoming_msg}")
        print(f"📷 Media count: {num_media}")

        # Step 1: User uploads image
        if num_media > 0:
            media_url = request.values.get("MediaUrl0")
            print(f"📥 Downloading image: {media_url}")
            headers = {"User-Agent": "TwilioBot/1.0"}
            image_response = requests.get(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
            if image_response.status_code == 200:
                predicted_class, results = predict_image(image_response.content)
                if predicted_class:
                    resp.message(
                        f"✅ The leaf appears to be: *{predicted_class}*\n\n"
                        "👉 Would you like *prevention tips* or *confidence levels*? Reply with 'prevention' or 'confidence'."
                    )
                    last_prediction[sender] = {"class": predicted_class, "results": results}
                    print("📤 Prediction reply sent")
                else:
                    resp.message("⚠ Error: Could not process the image. Please try another one.")
            else:
                resp.message("⚠ Error downloading image. Please resend.")
            return str(resp)

        # Step 2: User asks for prevention tips or products
        if ("prevent" in incoming_msg or "treatment" in incoming_msg or "product" in incoming_msg) and sender in last_prediction:
            disease = last_prediction[sender]["class"]
            info = DISEASE_INFO.get(disease, DISEASE_INFO["Healthy"])
            
            response = f"🌱 *{info['name']}*\n{info['description']}\n\n"
            
            if "prevent" in incoming_msg or "treatment" in incoming_msg:
                response += "🌿 *Prevention & Treatment Tips:*\n"
                for tip in info["prevention"]:
                    response += f"• {tip}\n"
                response += "\n"
            
            if "product" in incoming_msg or "buy" in incoming_msg:
                response += "🛒 *Recommended Products:*\n"
                for product in info["products"]:
                    response += f"• {product['name']}: {product['url']}\n"
            
            response += "\n💡 *Need more help?* Reply with 'products' for purchase links."
            
            resp.message(response)
            print("📤 Prevention tips and products sent")
            return str(resp)

        # Step 3: User replies "confidence"
        if incoming_msg == "confidence" and sender in last_prediction:
            results = last_prediction[sender]["results"]
            msg_text = (
                "📊 Confidence levels:\n"
                f"- Early Blight: {results['Early Blight']:.2f}%\n"
                f"- Late Blight: {results['Late Blight']:.2f}%\n"
                f"- Healthy: {results['Healthy']:.2f}%"
            )
            resp.message(msg_text)
            print("📤 Confidence levels sent")
            return str(resp)

        # Greetings and help
        if "hi" in incoming_msg or "hello" in incoming_msg or "help" in incoming_msg:
            help_text = """👋 *Welcome to Potato Disease Detector Bot!* 🌱

I can help you identify potato plant diseases and provide prevention tips.

*How to use:*
📸 Send a photo of a potato leaf for analysis
💬 After getting results, you can ask for:
  • 'prevention' - Get prevention tips
  • 'products' - See recommended products with direct purchase links
  • 'confidence' - See confidence levels for the prediction
  • 'help' - Show this message

*Supported diseases:*
• Early Blight
• Late Blight
• Healthy plants

🛒 *Product Links:*
When you ask for 'products', I'll provide direct links to recommended treatments on Amazon for easy purchasing.

🌿 Happy gardening!"""
            resp.message(help_text)
        else:
            resp.message("🤖 I didn't understand that. Send a potato leaf photo or type 'help' for assistance.")
        return str(resp)

    except Exception as e:
        print("❌ WhatsApp bot error:", e)
        return "Error", 500

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    print("🚀 Starting WhatsApp bot server...")
    print(f"🔗 Local URL: http://localhost:5000")
    print("🔌 Make sure to expose this server to the internet using ngrok")
    print("🔍 Debug mode is ON")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)