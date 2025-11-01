# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (only runtime, not build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies and clean up
RUN pip install --no-cache-dir -r requirements.txt && \
    # Remove unnecessary files from TensorFlow
    find /usr/local/lib/python3.12/site-packages/ -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages/ -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages/ -name "*.pyc" -delete && \
    find /usr/local/lib/python3.12/site-packages/ -name "*.pyo" -delete && \
    find /usr/local/lib/python3.12/site-packages/ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    # Remove pip cache
    rm -rf /root/.cache/pip

# Copy application code
COPY app.py .
COPY templates/ templates/
COPY static/ static/
COPY slides/ slides/
COPY models/ models/
COPY README.md .

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the application
CMD ["python", "app.py"]
