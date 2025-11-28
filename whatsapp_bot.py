import os
import io
import requests  
import numpy as np
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
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
print("🔍 Loading custom trained model...")
# Get the absolute path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'potato_disease_model.keras')
print(f"Looking for model at: {model_path}")
model = load_model(model_path)
print("✅ Model loaded successfully!")

# Store the last prediction for each user
last_prediction = {}

# Disease information
DISEASE_INFO = {
    "Early Blight": {
        "name": "Early Blight",
        "description": "Early blight is a common fungal disease that affects potato plants, causing dark spots with concentric rings on leaves.",
        "prevention": [
            "Rotate crops every 2-3 years",
            "Remove and destroy infected plant debris",
            "Water at the base of plants to keep foliage dry",
            "Apply fungicides preventatively during wet weather"
        ],
        "products": [
            {"name": "Copper Fungicide Spray", "url": "https://example.com/copper-fungicide"},
            {"name": "Chlorothalonil Fungicide", "url": "https://example.com/chlorothalonil"},
            {"name": "Mancozeb Fungicide", "url": "https://example.com/mancozeb"}
        ],
        "buy_links": [
            "🔗 Amazon: https://www.amazon.com/s?k=copper+fungicide+for+plants",
            "🔗 Home Depot: https://www.homedepot.com/s/copper%2520fungicide",
            "🔗 Local garden centers"
        ]
    },
    "Late Blight": {
        "name": "Late Blight",
        "description": "A serious disease caused by Phytophthora infestans, leading to rapid plant destruction if not controlled.",
        "prevention": [
            "Plant certified disease-free seed potatoes",
            "Ensure good air circulation between plants",
            "Remove and destroy infected plants immediately",
            "Apply fungicides before disease appears"
        ],
        "products": [
            {"name": "Phytophthora Fungicide", "url": "https://example.com/phytophthora-fungicide"},
            {"name": "Copper Fungal Treatment", "url": "https://example.com/copper-treatment"},
            {"name": "Systemic Fungicide", "url": "https://example.com/systemic-fungicide"}
        ],
        "buy_links": [
            "🔗 Amazon: https://www.amazon.com/s?k=late+blight+fungicide",
            "🔗 Lowe's: https://www.lowes.com/search?searchTerm=plant+fungicide",
            "🔗 Agricultural supply stores"
        ]
    },
    "Healthy": {
        "name": "Healthy",
        "description": "Your potato plant appears to be healthy! Continue with good cultural practices to maintain plant health.",
        "prevention": [
            "Use balanced fertilizer (10-10-10 NPK)",
            "Maintain consistent soil moisture",
            "Monitor for pests regularly",
            "Practice crop rotation"
        ],
        "products": [
            {"name": "Balanced NPK Fertilizer", "url": "https://example.com/npk-fertilizer"},
            {"name": "Organic Compost", "url": "https://example.com/organic-compost"},
            {"name": "Plant Vitamins", "url": "https://example.com/plant-vitamins"}
        ],
        "buy_links": [
            "🔗 Amazon: https://www.amazon.com/s?k=organic+plant+fertilizer",
            "🔗 Local garden centers",
            "🔗 Home improvement stores"
        ]
    }
}

def predict_image(image_bytes):
    try:
        # Load and preprocess the image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = ImageOps.fit(img, (128, 128), Image.Resampling.LANCZOS)  # Match training size
        img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Map predictions to class names and format results
        results = {
            "Early Blight": float(predictions[0]) * 100,
            "Late Blight": float(predictions[1]) * 100,
            "Healthy": float(predictions[2]) * 100
        }
        
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
                    if isinstance(product, dict):
                        response += f"• [{product['name']}]({product['url']})\n"
                    else:
                        response += f"• {product}\n"
                
                response += "\n🌐 *Where to Buy Products:*\n"
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