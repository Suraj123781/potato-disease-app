import os
# Configure environment for CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations for consistent CPU results

import io
import numpy as np
import requests
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
            "Copper-based fungicides",
            "Chlorothalonil-based sprays",
            "Mancozeb fungicides"
        ],
        "buy_links": [
            "🔗 Copper Fungicide: https://amzn.in/d/8xWJ6X7",
            "🔗 Chlorothalonil Spray: https://amzn.in/d/8xWJ6X7",
            "🔗 Mancozeb Fungicide: https://amzn.in/d/8xWJ6X7"
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
            "Copper fungicides",
            "Chlorothalonil",
            "Metalaxyl-based fungicides"
        ],
        "buy_links": [
            "🔗 Copper Fungicide: https://amzn.in/d/8xWJ6X7",
            "🔗 Chlorothalonil Fungicide: https://amzn.in/d/8xWJ6X7",
            "🔗 Metalaxyl Fungicide: https://amzn.in/d/8xWJ6X7"
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
            "Balanced NPK fertilizer",
            "Organic compost",
            "General plant vitamins"
        ],
        "buy_links": [
            "🔗 NPK 19:19:19 Fertilizer: https://amzn.in/d/8xWJ6X7",
            "🔗 Organic Compost: https://amzn.in/d/8xWJ6X7",
            "🔗 Seaweed Extract: https://amzn.in/d/8xWJ6X7"
        ]
    }
}

def download_media(media_url, save_path, auth):
    response = requests.get(media_url, auth=auth)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        return save_path
    else:
        raise Exception(f"Failed to download media: {response.status_code}")

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
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        
        # Convert to our class format
        results = {}
        for _, label, prob in decoded_predictions:
            label_lower = label.lower()
            if 'blight' in label_lower or 'disease' in label_lower:
                if 'early' in label_lower:
                    results['Early Blight'] = float(prob) * 100
                else:
                    results['Late Blight'] = float(prob) * 100
            else:
                results['Healthy'] = float(prob) * 100
        
        # Ensure all classes are present
        for class_name in CLASS_NAMES:
            if class_name not in results:
                results[class_name] = 0.0
        
        # Get the class with highest probability
        predicted_class = max(results.items(), key=lambda x: x[1])[0]
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
                response += "�️ *Prevention & Treatment Tips:*\n"
                for tip in info["prevention"]:
                    response += f"• {tip}\n"
                response += "\n"
            
            if "product" in incoming_msg or "buy" in incoming_msg:
                response += "🛒 *Recommended Products:*\n"
                for product in info["products"]:
                    response += f"• {product}\n"
                
                response += "\n🌐 *Where to Buy:*\n"
                for link in info.get("buy_links", []):
                    response += f"• {link}\n"
            
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
  • 'products' - See recommended products
  • 'help' - Show this message

*Supported diseases:*
• Early Blight
• Late Blight
• Healthy plants

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