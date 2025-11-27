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

# Prevention tips + shopping links
SUGGESTIONS = {
    "Early Blight": (
        "🛡 Use fungicides like chlorothalonil or mancozeb. Remove infected leaves and rotate crops.\n\n"
        "🛒 Buy online:\n"
        "- AgriBegri: https://agribegri.com/products/shivalik-zee-l-fungicide.php\n"
        "- BigHaat: https://www.bighaat.com/collections/management-of-early-blight-in-tomato-and-potato\n"
        "- Amazon: https://www.amazon.in/Blitox-RALLIS-Copper-Oxychloride-Fungicide/dp/B0CKW9LGL1\n"
        "- UPL Blitox (Copper Oxychloride) via AgriBegri: https://agribegri.com/products/blitox-fungicide.php"


    ),
    "Late Blight": (
        "🧪 Apply copper-based fungicides. Avoid overhead watering and improve air circulation.\n\n"
        "🛒 Buy online:\n"
        "- BharatAgri: https://krushidukan.bharatagri.com/en/collections/late-blight-disease-products-online\n"
        "- BigHaat: https://www.bighaat.com/collections/late-blight-disease-management-in-tomato-and-potato-crops\n"
        "- AgriBegri: https://agribegri.com/en/products/buy-sumitomo-kemoxyl-metalaxyl-8--mancozeb-64-wp-fungicide-online.php\n"
        "- Amazon: https://www.amazon.in/Katyayani-Blight-Metalaxyl-M-Chlorothalonil-Fast-Acting/dp/B0FT3TQX58"

    ),
    "Healthy": "✅ No action needed. Maintain regular monitoring and good soil health."
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

        # Step 2: User replies "prevention"
        if incoming_msg == "prevention" and sender in last_prediction:
            disease = last_prediction[sender]["class"]
            resp.message(f"💡 Prevention tips for *{disease}*:\n\n{SUGGESTIONS[disease]}")
            print("📤 Prevention tips sent")
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

        # Greetings and fallback
        if "hi" in incoming_msg or "hello" in incoming_msg:
            resp.message("👋 Hello! Send me a *potato leaf image*, and I'll tell you if it's *Early Blight*, *Late Blight*, or *Healthy*. 🌿")
        else:
            resp.message("🤖 I didn't understand that. Send a leaf image or say 'hi'.")
        return str(resp)

    except Exception as e:
        print("❌ WhatsApp bot error:", e)
        return "Error", 500

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)