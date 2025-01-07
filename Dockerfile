# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required for RDKit
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libboost-all-dev \
    cmake \
    libxrender1 \
    libxext6 \
    libxft2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY fluroml-app.py .
COPY best_classifier_compatible.joblib .
COPY new_best_regressor_compatible.joblib .
COPY best_regressor_emission_compatible.joblib .

# Expose Streamlit's default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "fluroml-app.py", "--server.port=8501", "--server.enableXsrfProtection=false"]
