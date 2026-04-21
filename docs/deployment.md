# Deployment Guide

## Local Deployment

### Prerequisites
- Python 3.8+
- 4GB RAM (8GB+ recommended)
- 2GB disk space

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/pneumonia-detection-ai.git
cd pneumonia-detection-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python run_app.py
```

Access at `http://localhost:8501`

---

## Docker Deployment

### Build Image

```bash
docker build -t pneumonia-detection:latest .
```

### Run Container

```bash
docker run -p 8501:8501 pneumonia-detection:latest
```

### Production Deployment with Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  pneumonia-detection:
    build: .
    ports:
      - "8501:8501"
    environment:
      - TF_CPP_MIN_LOG_LEVEL=2
      - STREAMLIT_SERVER_HEADLESS=true
    volumes:
      - ./app/model:/app/app/model
      - ./xray_dataset:/app/xray_dataset
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Run with:
```bash
docker-compose up -d
```

---

## Cloud Deployment

### AWS EC2

```bash
# Launch EC2 instance (t3.large or larger)
# Connect to instance

# Install dependencies
sudo yum update -y
sudo yum install python3-pip git -y

# Clone and setup
git clone https://github.com/yourusername/pneumonia-detection-ai.git
cd pneumonia-detection-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install and run
pip install -r requirements.txt
streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
```

### Heroku

```bash
# Create Heroku app
heroku create pneumonia-detection-ai

# Set buildpacks
heroku buildpacks:add heroku/python

# Deploy
git push heroku main

# View logs
heroku logs --tail
```

### Google Cloud Run

```bash
# Authenticate
gcloud auth login

# Deploy
gcloud run deploy pneumonia-detection \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --timeout 3600
```

---

## Security Considerations

1. **Model Protection**
   - Store model weights securely
   - Use encrypted storage for sensitive files
   - Implement access controls

2. **Data Privacy**
   - Do not log patient data
   - Use HTTPS for all connections
   - Implement data retention policies

3. **Medical Compliance**
   - Add audit logging
   - Implement user authentication
   - Maintain access records

---

## Monitoring & Logging

### Enable Logging

```python
from app.logger import setup_logger

logger = setup_logger(__name__)
logger.info("Application started")
```

### Health Checks

```bash
curl http://localhost:8501/_stcore/health
```

---

## Environment Variables

```bash
# Model configuration
MODEL_PATH="app/model/resnet_model.h5"

# Streamlit settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost

# TensorFlow optimization
TF_CPP_MIN_LOG_LEVEL=2
TF_ENABLE_ONEDNN_OPTS=0

# Logging
LOG_LEVEL=INFO
```

---

## Scaling

### Horizontal Scaling
- Use load balancer (NGINX, HAProxy)
- Deploy multiple instances
- Implement session persistence

### Vertical Scaling
- Increase server resources
- Enable GPU support
- Optimize model inference

---

## Troubleshooting

### Model Not Found
```bash
# Verify model location
ls -la app/model/

# Download model if missing
python scripts/download_model.py
```

### Memory Issues
```bash
# Reduce batch size in config.py
BATCH_SIZE = 16

# Enable model quantization
```

### Slow Inference
```bash
# Enable GPU
# Install CUDA-compatible TensorFlow
pip install tensorflow[and-cuda]

# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## Maintenance

### Regular Updates
```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Check security vulnerabilities
pip-audit
```

### Backup Strategy
- Back up model weights daily
- Archive logs weekly
- Maintain database backups

### Performance Tuning
- Monitor inference time
- Optimize database queries
- Cache frequently used results
