# Multi-stage Dockerfile for Pneumonia Detection AI

# Stage 1: Base image with dependencies
FROM python:3.10-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Build dependencies
FROM base as builder

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 3: Final production image
FROM base

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Set environment variables
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0

# Copy application code
COPY app/ ./app/
COPY run_app.py .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose Streamlit port
EXPOSE 8501

# Run the application
ENTRYPOINT ["python", "run_app.py"]
