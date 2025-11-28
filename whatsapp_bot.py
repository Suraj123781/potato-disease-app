import os
import io
import numpy as np
import logging
from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Twilio credentials
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')

# Configuration
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'potato_disease_model.keras')

# Initialize model
model = None

def load_model():
    """Load the trained potato disease classification model."""
    global model
    try:
        logger.info(f"🔍 Loading model from: {MODEL_PATH}")
        
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        # Load the model
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        logger.info("✅ Model loaded successfully!")
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shape: {model.output_shape}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        return False

# Load model when the app starts
if not load_model():
    logger.error("Failed to load model. Exiting...")
    exit(1)

# Store the last prediction for each user
last_prediction = {}

# Disease information
DISEASE_INFO = {
    "Early Blight": {
        "name": "Early Blight",
        "description": "Early blight is a common fungal disease that affects potato plants, causing dark spots with concentric rings on leaves.",
        "symptoms": [
            "Small, dark brown to black spots on lower leaves",
            "Concentric rings in the spots (target-like appearance)",
            "Yellowing of leaves around the spots",
            "Premature leaf drop in severe cases"
        ],
        "prevention": [
            "Rotate crops every 2-3 years with non-solanaceous plants",
            "Remove and destroy infected plant debris after harvest",
            "Use certified disease-free seed potatoes",
            "Water at the base of plants to keep foliage dry",
            "Apply fungicides preventatively during wet weather"
        ],
        "treatment": [
            "Apply copper-based fungicides at first sign of disease",
            "Use chlorothalonil or mancozeb-based fungicides",
            "Remove and destroy severely infected plants",
            "Improve air circulation around plants"
        ],
        "products": [
            "Copper Fungicide Spray (e.g., Bonide Copper Fungicide)",
            "Chlorothalonil-based fungicides (e.g., Daconil)",
            "Mancozeb-based fungicides"
        ],
        "buy_links": [
            "🔗 Amazon: https://www.amazon.com/s?k=copper+fungicide+for+plants",
            "🔗 Home Depot: https://www.homedepot.com/s/copper%2520fungicide",
            "🔗 Local garden centers"
        ]
    },
    "Late Blight": {
        "name": "Late Blight",
        "description": "A serious disease caused by Phytophthora infestans, leading to rapid plant destruction if not controlled. It was responsible for the Irish Potato Famine.",
        "symptoms": [
            "Water-soaked spots on leaves that turn brown and papery",
            "White fungal growth under leaves in humid conditions",
            "Dark, greasy-looking lesions on stems",
            "Rapid spread in cool, wet weather"
        ],
        "prevention": [
            "Plant certified disease-free seed potatoes",
            "Choose resistant varieties when available",
            "Space plants properly for good air circulation",
            "Avoid overhead watering",
            "Apply preventive fungicides in high-risk periods"
        ],
        "treatment": [
            "Apply fungicides containing chlorothalonil or mancozeb",
            "Use systemic fungicides for active infections",
            "Remove and destroy infected plants immediately",
            "Avoid working in wet fields to prevent spread"
        ],
        "products": [
            "Phytophthora Fungicide (e.g., Agri-Fos, Fosphite)",
            "Copper Fungal Treatment",
            "Systemic Fungicide (e.g., Revus, Ranman)"
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
        "care_tips": [
            "Use balanced fertilizer (10-10-10 NPK) every 4-6 weeks",
            "Maintain consistent soil moisture (about 1-2 inches per week)",
            "Monitor for pests like Colorado potato beetles",
            "Hill soil around plants as they grow to prevent greening of tubers"
        ],
        "prevention": [
            "Practice 3-year crop rotation",
            "Use certified disease-free seed potatoes",
            "Keep garden free of weeds and plant debris",
            "Monitor plants weekly for early signs of disease"
        ],
        "products": [
            "Balanced NPK Fertilizer (10-10-10)",
            "Organic Compost",
            "Plant Vitamins (e.g., fish emulsion, seaweed extract)",
            "Mulch (straw or shredded leaves)"
        ],
        "buy_links": [
            "🔗 Amazon: https://www.amazon.com/s?k=organic+plant+fertilizer",
            "🔗 Local garden centers",
            "🔗 Home improvement stores"
        ]
    }
}

def preprocess_image(image_bytes):
    """Preprocess the image for model prediction."""
    try:
        # Open and resize image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to array and preprocess
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_disease(image_bytes):
    """Predict the disease from an image."""
    try:
        # Preprocess the image
        img_array = preprocess_image(image_bytes)
        if img_array is None:
            return None, {class_name: 0.0 for class_name in CLASS_NAMES}
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Convert to percentage and create results dictionary
        results = {class_name: float(prob) * 100 
                  for class_name, prob in zip(CLASS_NAMES, predictions)}
        
        # Get the class with highest probability
        predicted_class = max(results.items(), key=lambda x: x[1])[0]
        
        logger.info(f"Prediction: {predicted_class} | {results}")
        return predicted_class, results
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return None, {class_name: 0.0 for class_name in CLASS_NAMES}

def format_response(predicted_class, confidence, include_products=False):
    """Format the response message with disease information."""
    info = DISEASE_INFO[predicted_class]
    confidence_percent = confidence[predicted_class]
    
    # Start building response
    response_parts = [
        f"🍁 *{info['name']} Detected* ({confidence_percent:.1f}% confidence)\n\n",
        f"📝 *Description:* {info['description']}\n\n"
    ]
    
    # Add symptoms if available
    if 'symptoms' in info:
        response_parts.append("🔍 *Symptoms:*\n")
        response_parts.extend(f"• {symptom}\n" for symptom in info['symptoms'])
        response_parts.append("\n")
    
    # Add prevention tips
    response_parts.append("🛡️ *Prevention Tips:*\n")
    response_parts.extend(f"• {tip}\n" for tip in info['prevention'])
    
    # Add treatment if available
    if 'treatment' in info and info['name'] != 'Healthy':
        response_parts.append("\n💊 *Treatment Options:*\n")
        response_parts.extend(f"• {treatment}\n" for treatment in info['treatment'])
    
    # Add products if requested
    if include_products and 'products' in info:
        response_parts.append("\n🛒 *Recommended Products:*\n")
        response_parts.extend(f"• {product}\n" for product in info['products'])
        
        if 'buy_links' in info and info['buy_links']:
            response_parts.append("\n🌐 *Where to Buy:*\n")
            response_parts.extend(f"{link}\n" for link in info['buy_links'])
    
    # Add care tips for healthy plants
    if info['name'] == 'Healthy' and 'care_tips' in info:
        response_parts.append("\n🌱 *Care Tips:*\n")
        response_parts.extend(f"• {tip}\n" for tip in info['care_tips'])
    
    return "".join(response_parts)

@app.route("/webhook", methods=['POST'])
def webhook():
    """Handle incoming WhatsApp messages."""
    # Get incoming message
    incoming_msg = request.values.get('Body', '').lower()
    phone_number = request.values.get('From', '')
    
    resp = MessagingResponse()
    msg = resp.message()
    
    logger.info(f"Incoming message from {phone_number}: {incoming_msg}")
    
    # Check if message contains media
    if request.values.get('NumMedia') != '0':
        try:
            # Get the image URL
            image_url = request.values.get('MediaUrl0')
            logger.info(f"Processing image from: {image_url}")
            
            # Download the image
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # Make prediction
            predicted_class, confidence = predict_disease(response.content)
            
            if predicted_class:
                # Store the last prediction for this user
                last_prediction[phone_number] = (predicted_class, confidence)
                
                # Format and send response
                include_products = any(word in incoming_msg for word in ['product', 'buy', 'treatment', 'cure'])
                response_msg = format_response(predicted_class, confidence, include_products)
                
                # Add follow-up prompt
                if predicted_class != 'Healthy':
                    response_msg += "\n💡 Reply 'products' for treatment recommendations or send another photo."
                else:
                    response_msg += "\n📸 Send another photo anytime to check your plants!"
                
                msg.body(response_msg)
                logger.info(f"Sent prediction: {predicted_class} to {phone_number}")
            else:
                msg.body("❌ Sorry, I couldn't process the image. Please try with a clearer photo of potato leaves.")
                logger.warning(f"Failed to process image from {phone_number}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading image: {str(e)}")
            msg.body("❌ Couldn't download the image. Please try sending it again.")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            msg.body("❌ An error occurred while processing your request. Please try again later.")
    
    # Handle text messages
    else:
        if "hi" in incoming_msg or "hello" in incoming_msg:
            welcome_msg = (
                "👋 *Hello! I'm your Potato Disease Detector* 🥔\n\n"
                "📸 Send me a clear photo of potato leaves, and I'll help identify any diseases.\n\n"
                "💡 *Tips for best results:*\n"
                "• Take photos in good lighting\n"
                "• Focus on the leaves\n"
                "• Avoid blurry or dark images\n\n"
                "Type 'help' for more options."
            )
            msg.body(welcome_msg)
            
        elif "product" in incoming_msg or "buy" in incoming_msg or "treatment" in incoming_msg:
            if phone_number in last_prediction:
                predicted_class, confidence = last_prediction[phone_number]
                response_msg = format_response(predicted_class, confidence, include_products=True)
                msg.body(response_msg)
            else:
                msg.body("ℹ️ Please send a photo of your potato plant first so I can recommend the right products.")
                
        elif "help" in incoming_msg:
            help_msg = (
                "ℹ️ *Potato Disease Detector Help*\n\n"
                "*Basic Commands:*\n"
                "- Send a photo of potato leaves to detect diseases\n"
                "- Type 'products' after detection to see recommended treatments\n"
                "- Type 'help' to see this message\n"
                "- Type 'about' to learn more about this bot\n\n"
                "*Diseases Detected:*\n"
                "- Early Blight\n"
                "- Late Blight\n"
                "- Healthy plants"
            )
            msg.body(help_msg)
            
        elif "about" in incoming_msg:
            about_msg = (
                "🌱 *About Potato Disease Detector*\n\n"
                "This AI-powered bot helps identify common potato plant diseases. "
                "It uses deep learning to analyze photos of potato leaves and provides "
                "information about detected diseases, prevention tips, and treatment options.\n\n"
                "*For best results:*\n"
                "• Take photos in natural light\n"
                "• Focus on affected leaves\n"
                "• Avoid shadows and glare\n\n"
                "This is an AI assistant and not a substitute for professional agricultural advice."
            )
            msg.body(about_msg)
            
        else:
            try:
                help_text = (
                    " *Potato Disease Detector Help* \n\n"
                    "Send me a photo of potato leaves, and I'll help identify any diseases.\n\n"
                    "*Available Commands:*\n"
                    "- help: Show this help message\n"
                    "- about: Learn more about this bot\n"
                    "- products: Show recommended products (after detection)\n\n"
                    "*Diseases Detected:*\n"
                    "- Early Blight\n"
                    "- Late Blight\n"
                )
                msg.body(help_text)
            except Exception as e:
                logger.error(f"Error sending help message: {str(e)}")
                msg.body("Sorry, I encountered an error while processing your request. Please try again later.")
                
            print("❌ WhatsApp bot error:", e)
            return "Error", 500

        return str(resp)

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    print("🚀 Starting WhatsApp bot server...")
    print(f"🔗 Local URL: http://localhost:5000")
    print("🔌 Make sure to expose this server to the internet using ngrok")
    print("🔍 Debug mode is ON")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)