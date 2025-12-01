import os
import io
import requests   # <-- critical import
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load environment variables
load_dotenv()
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

print("🔑 SID:", TWILIO_ACCOUNT_SID)
print("🔑 TOKEN:", "Loaded" if TWILIO_AUTH_TOKEN else "Missing")

app = Flask(__name__)

# -----------------------------
# Safe Model Load
# -----------------------------
try:
    model = tf.keras.models.load_model("potato_disease_model.keras", compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model load failed:", e)
    model = None

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

SUGGESTIONS = {
    "Early Blight": (
        "🛡 Use fungicides like chlorothalonil or mancozeb. Remove infected leaves and rotate crops.\n\n"
        "🛒 Buy online:\n"
        "- AgriBegri: https://agribegri.com/products/shivalik-zee-l-fungicide.php\n"
        "- BigHaat: https://www.bighaat.com/collections/management-of-early-blight-in-tomato-and-potato\n"
        "- Amazon: https://www.amazon.in/Blitox-RALLIS-Copper-Oxychloride-Fungicide/dp/B0CKW9LGL1\n"
        "- UPL Blitox via AgriBegri: https://agribegri.com/products/blitox-fungicide.php"
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

# Store last prediction per user
last_prediction = {}

def predict_image(image_bytes):
    if model is None:
        print("❌ Model not available")
        return None, None
    try:
        # 1. Open and convert image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print(f"✅ Loaded image with size: {img.size} and mode: {img.mode}")
        
        # 2. Try multiple input sizes
        input_sizes = [(224, 224), (128, 128), (256, 256), (150, 150)]
        best_confidence = 0
        best_result = None
        best_class = None
        best_size = None
        
        for target_size in input_sizes:
            try:
                # 3. Resize and preprocess
                img_resized = img.resize(target_size)
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # 4. Make prediction
                predictions = model.predict(img_array, verbose=0)[0]
                print(f"Raw predictions for {target_size}: {predictions}")
                
                # 5. Get class with highest probability
                predicted_idx = np.argmax(predictions)
                confidence = predictions[predicted_idx]
                predicted_class = CLASS_NAMES[predicted_idx]
                
                # 6. Store best prediction
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_class = predicted_class
                    best_size = target_size
                    best_result = {
                        "Early Blight": float(predictions[0]) * 100,
                        "Late Blight": float(predictions[1]) * 100,
                        "Healthy": float(predictions[2]) * 100
                    }
                    
                print(f"Size {target_size}: {predicted_class} ({confidence:.2%})")
                
            except Exception as e:
                print(f"⚠️ Error with size {target_size}: {e}")
                continue
        
        # 7. Check if we got any predictions
        if best_result is None:
            return "Error - Could not process image", None
            
        print(f"\n🎯 Best prediction: {best_class} (Confidence: {best_confidence:.2%} at size {best_size})")
        print(f"All confidences: {best_result}")
        
        # 8. Return the best prediction with detailed results
        return best_class, best_result
        
    except Exception as e:
        error_msg = f"❌ Error in predict_image: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, {"Error": error_msg}
        
@app.route("/", methods=["GET"])
def home():
    return "✅ Potato Leaf Disease Classifier is running.", 200

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

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

            try:
                image_response = requests.get(
                    media_url,
                    auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
                    timeout=10
                )
                print(f"📦 Image download status: {image_response.status_code}")

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
                    resp.message(f"⚠ Error downloading image. Status: {image_response.status_code}")
                print("🔧 TwiML response:", str(resp))
                return str(resp)

            except Exception as e:
                print("❌ Exception while downloading:", e)
                resp.message(f"⚠ Error downloading image: {e}")
                return str(resp)

        # Step 2: User replies "prevention"
        if incoming_msg == "prevention" and sender in last_prediction:
            disease = last_prediction[sender]["class"]
            resp.message(f"💡 Prevention tips for *{disease}*:\n\n{SUGGESTIONS[disease]}")
            print("📤 Prevention tips sent")
            print("🔧 TwiML response:", str(resp))
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
            print("🔧 TwiML response:", str(resp))
            return str(resp)

        # Step 4: Greetings and fallback
        if "hi" in incoming_msg or "hello" in incoming_msg:
            resp.message("👋 Hello! Send me a *potato leaf image*, and I'll tell you if it's *Early Blight*, *Late Blight*, or *Healthy*. 🌿")
            print("📤 Greeting reply sent")
            print("🔧 TwiML response:", str(resp))
            return str(resp)

        resp.message("🤖 I didn't understand that. Send a leaf image or say 'hi'.")
        print("📤 Fallback reply sent")
        print("🔧 TwiML response:", str(resp))
        return str(resp)

    except Exception as e:
        print("❌ WhatsApp bot error:", e)
        return "Error", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)