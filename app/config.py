"""
Configuration and constants for the Pneumonia Detection AI application.

This module centralizes all configuration settings, paths, and constants
used across the application.
"""

from pathlib import Path
from typing import Final

# ==================== PROJECT STRUCTURE ====================
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
APP_DIR: Final[Path] = PROJECT_ROOT / "app"
MODEL_DIR: Final[Path] = APP_DIR / "model"
DATA_DIR: Final[Path] = PROJECT_ROOT / "xray_dataset"
SAMPLE_DIR: Final[Path] = APP_DIR / "sample"

# ==================== MODEL CONFIGURATION ====================
MODEL_PATH: Final[Path] = MODEL_DIR / "resnet_model.h5"
INPUT_SIZE: Final[tuple] = (256, 256)
BATCH_SIZE: Final[int] = 32
CONFIDENCE_THRESHOLD: Final[float] = 0.5

# ==================== PERFORMANCE METRICS ====================
# These are baseline metrics from training evaluation
MODEL_METRICS = {
    "accuracy": 0.95,
    "sensitivity": 0.96,
    "specificity": 0.94,
    "precision": 0.94,
    "recall": 0.96,
    "f1_score": 0.95,
    "auc_roc": 0.98,
    "training_samples": 5800,
    "test_samples": 600,
}

# ==================== UI CONFIGURATION ====================
PAGE_CONFIG = {
    "page_title": "Pneumonia Detection AI",
    "page_icon": "🫁",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# ==================== COLOR SCHEME ====================
COLORS = {
    "primary": "#0ea5e9",
    "primary_dark": "#0284c7",
    "primary_light": "#e0f2fe",
    "accent": "#10b981",
    "accent_dark": "#047857",
    "success": "#16a34a",
    "danger": "#dc2626",
    "warning": "#f59e0b",
    "info": "#3b82f6",
}

# ==================== CLASSES ====================
CLASS_LABELS = {
    0: "NORMAL",
    1: "PNEUMONIA",
}

CLASS_DESCRIPTIONS = {
    "NORMAL": "✅ Healthy - No pneumonia detected",
    "PNEUMONIA": "⚠️ Abnormal - Pneumonia infection likely",
}

# ==================== FILE LIMITS ====================
MAX_UPLOAD_SIZE_MB: Final[int] = 25
SUPPORTED_FORMATS: Final[tuple] = ("jpg", "jpeg", "png", "gif", "bmp")

# ==================== 3D VISUALIZATION ====================
HEIGHT_SCALE: Final[float] = 35.0
DOWNSAMPLE_FACTOR: Final[int] = 2
HOVER_TEMPLATE: Final[str] = (
    "<b>📍 Position</b><br>"
    "X: %{x:.0f}<br>"
    "Y: %{y:.0f}<br>"
    "Z: %{z:.1f}<br>"
    "<b>🔍 Attention: %{surfacecolor:.1%}</b><br>"
    "<b>🎯 Risk Level: %{surfacecolor:.0%}</b>"
    "<extra></extra>"
)

# ==================== API & LOGGING ====================
LOG_LEVEL: Final[str] = "INFO"
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ==================== DOCUMENTATION URLS ====================
DOCUMENTATION = {
    "model_paper": "https://example.com/model-paper",
    "dataset_source": "https://example.com/chest-xray-dataset",
    "research_base": "https://example.com/research",
}

# ==================== DISCLAIMERS ====================
MEDICAL_DISCLAIMER = (
    "⚠️ **IMPORTANT DISCLAIMER**\n\n"
    "This AI system is designed for **educational and research purposes only**. "
    "It is **NOT** a substitute for professional medical diagnosis. "
    "Results must be reviewed by qualified healthcare professionals. "
    "Always consult with a physician before making any medical decisions."
)

MODEL_LIMITATIONS = [
    "Trained on limited dataset - may not generalize to all populations",
    "Works best with frontal chest X-rays",
    "May have lower accuracy on image artifacts or quality issues",
    "Cannot detect all types of pneumonia",
    "Should be used as a screening tool, not for final diagnosis",
]
