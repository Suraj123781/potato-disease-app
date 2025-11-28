# Use Python 3.10 slim image for smaller size
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files first to leverage Docker cache
COPY app.py whatsapp_bot.py requirements.txt ./

# Copy model files
COPY potato_disease_model.keras best_model.keras potato_disease_model.tflite ./

# Verify the model file is present
RUN echo "Verifying model files exist..." && \
    if [ ! -f "potato_disease_model.keras" ]; then \
        echo "Error: Model file not found!" && \
        echo "Current directory contents:" && \
        ls -la && \
        exit 1; \
    fi && \
    echo "Model files verified successfully"

# Expose the port the app runs on
EXPOSE $PORT

# Command to run the application
CMD ["waitress-serve", "--port=$PORT", "whatsapp_bot:app"]
