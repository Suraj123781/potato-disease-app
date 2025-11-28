# Use Python 3.10 as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model file
COPY potato_disease_model.keras .

# Copy the application code
COPY whatsapp_bot.py .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["waitress-serve", "--port=5000", "whatsapp_bot:app"]
