import os
import io
import requests
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
load_dotenv()
app = Flask(__name__)

# -----------------------------
# Safe Model Load
# -----------------------------
try:
    model = tf.keras.models.load_model("potato_disease_model.keras")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model load failed:", e)
    model = None  # fallback so app still runs

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Disease information and prevention tips
DISEASE_INFO = {
    "Early Blight": {
        "name": "Early Blight",
        "description": "Early blight is a common fungal disease that causes dark spots with concentric rings on leaves.",
        "prevention": [
            "🔄 Rotate crops (don't plant potatoes in the same place for 3-4 years)",
            "🌱 Use certified disease-free seed potatoes",
            "💧 Water at the base of plants to keep foliage dry",
            "🧹 Remove and destroy infected plant debris",
            "🌿 Apply mulch to prevent soil splashing onto leaves"
        ],
        "products": [
            "🔹 Copper Fungicide: https://amzn.to/3XbY5Qp",
            "🔹 Neem Oil: https://amzn.to/3x8Yr0S",
            "🔹 Disease-Resistant Varieties: https://amzn.to/3x8Yr0T",
            "🔹 Mancozeb Fungicide: https://amzn.to/3XbY5Qr"
        ],
        "buy_links": [
            "🛒 AgriBegri: https://agribegri.com/products/shivalik-zee-l-fungicide.php",
            "🛒 BigHaat: https://www.bighaat.com/collections/management-of-early-blight-in-tomato-and-potato",
            "🛒 Amazon: https://www.amazon.in/Blitox-RALLIS-Copper-Oxychloride-Fungicide/dp/B0CKW9LGL1"
        ]
    },
    "Late Blight": {
        "name": "Late Blight",
        "description": "Late blight is a serious disease that can destroy entire crops, causing water-soaked spots on leaves.",
        "prevention": [
            "💨 Ensure good air circulation between plants",
            "☀️ Water in the morning to allow leaves to dry",
            "⚠️ Remove and destroy infected plants immediately",
            "🌧️ Avoid overhead watering",
            "🌱 Use resistant varieties when possible"
        ],
        "products": [
            "🔹 Chlorothalonil Fungicide: https://amzn.to/3XbY5Qr",
            "🔹 Copper Fungicide: https://amzn.to/3XbY5Qp",
            "🔹 Metalaxyl-based fungicides"
        ],
        "buy_links": [
            "🛒 BharatAgri: https://krushidukan.bharatagri.com/en/collections/late-blight-disease-products-online",
            "🛒 BigHaat: https://www.bighaat.com/collections/late-blight-disease-management-in-tomato-and-potato-crops",
            "🛒 Amazon: https://www.amazon.in/Katyayani-Blight-Metalaxyl-M-Chlorothalonil-Fast-Acting/dp/B0FT3TQX58"
        ]
    },
    "Healthy": {
        "name": "Healthy Plant",
        "description": "Your potato plant appears to be healthy. Continue with good cultural practices.",
        "prevention": [
            "🔍 Monitor plants regularly for early signs of disease",
            "🌱 Maintain proper soil nutrition and pH",
            "💧 Water consistently but avoid overwatering",
            "🌿 Use organic mulch to retain moisture",
            "� Encourage beneficial insects"
        ],
        "products": [
            "🌱 Organic Fertilizer: https://amzn.to/3x8Yr0U",
            "🧪 Soil Test Kit: https://amzn.to/3x8Yr0V",
            "🌿 Compost Bin: https://amzn.to/3x8Yr0W"
        ],
        "buy_links": [
            "🛒 Buy organic fertilizers from Ugaoo: https://www.ugaoo.com/organic-fertilizers.html",
            "🛒 Get gardening tools on Amazon: https://www.amazon.in/gp/bestsellers/kitchen/1374445031"
        ]
    }
}

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

# Store last prediction per user
last_prediction = {}

def predict_image(image_bytes):
    if model is None:
        print("❌ Model not available")
        return None, None
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)[0]

        results = {CLASS_NAMES[i]: float(predictions[i]) * 100 for i in range(len(CLASS_NAMES))}
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        print(f"✅ Prediction: {predicted_class} | {results}")
        return predicted_class, results
    except Exception as e:
        print("❌ Error processing image:", e)
        return None, None

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
    app.run(host="0.0.0.0", port=5000)