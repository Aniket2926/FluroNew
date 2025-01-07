FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the project files
COPY . .

# Install system dependencies for RDKit
RUN apt-get update && apt-get install -y \
    libboost-all-dev \
    libcairo2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for Streamlit
EXPOSE 8501

# Start the Streamlit app
CMD ["streamlit", "run", "fluroml-app.py", "--server.port=8501", "--server.address=0.0.0.0"]
