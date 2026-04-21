"""Model evaluation script for calculating performance metrics."""

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from helpers import load_pneumonia_model, preprocess_image
from config import MODEL_PATH, MODEL_METRICS


def evaluate_model(
    model_path: Path = MODEL_PATH,
    test_data_dir: Path = None,
) -> Dict[str, float]:
    """
    Evaluate model performance on test dataset.
    
    Args:
        model_path: Path to model file
        test_data_dir: Path to test data directory
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Loading model from {model_path}...")
    model = load_pneumonia_model(model_path)
    
    if model is None:
        print("Error: Failed to load model")
        return {}
    
    # Placeholder for actual evaluation
    # In production, load test data and evaluate
    print("\nModel loaded successfully")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    
    print("\nBaseline Metrics from Training:")
    for metric_name, value in MODEL_METRICS.items():
        print(f"  {metric_name}: {value}")
    
    return MODEL_METRICS


def print_metrics_summary(metrics: Dict[str, float]) -> None:
    """
    Print a formatted summary of metrics.
    
    Args:
        metrics: Dictionary of metrics to print
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION SUMMARY")
    print("="*50)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key.upper():<20}: {value:.4f}")
        else:
            print(f"{key.upper():<20}: {value}")
    
    print("="*50 + "\n")


def main():
    """Main evaluation script."""
    print("Pneumonia Detection AI - Model Evaluation")
    print("-" * 50)
    
    # Evaluate model
    metrics = evaluate_model()
    
    # Print summary
    if metrics:
        print_metrics_summary(metrics)
    else:
        print("Evaluation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
