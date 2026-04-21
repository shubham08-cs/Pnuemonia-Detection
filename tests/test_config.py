"""Tests for configuration module."""

import sys
from pathlib import Path
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from config import (
    PROJECT_ROOT,
    APP_DIR,
    MODEL_DIR,
    DATA_DIR,
    CLASS_LABELS,
    MODEL_METRICS,
    COLORS,
    SUPPORTED_FORMATS,
)


class TestProjectStructure:
    """Test project directory structure."""

    def test_project_root_exists(self):
        """Test that project root is correctly set."""
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()

    def test_app_dir_exists(self):
        """Test that app directory exists."""
        assert APP_DIR.exists()
        assert APP_DIR.is_dir()

    def test_app_dir_is_child_of_project(self):
        """Test that app dir is inside project root."""
        assert APP_DIR.parent == PROJECT_ROOT


class TestConfiguration:
    """Test configuration values."""

    def test_class_labels_format(self):
        """Test that class labels are properly defined."""
        assert isinstance(CLASS_LABELS, dict)
        assert 0 in CLASS_LABELS
        assert 1 in CLASS_LABELS
        assert CLASS_LABELS[0] == "NORMAL"
        assert CLASS_LABELS[1] == "PNEUMONIA"

    def test_model_metrics_exist(self):
        """Test that model metrics are defined."""
        assert isinstance(MODEL_METRICS, dict)
        assert "accuracy" in MODEL_METRICS
        assert "sensitivity" in MODEL_METRICS
        assert MODEL_METRICS["accuracy"] > 0.9

    def test_colors_defined(self):
        """Test that color scheme is defined."""
        assert isinstance(COLORS, dict)
        assert "primary" in COLORS
        assert "danger" in COLORS
        # Colors should be hex format
        assert COLORS["primary"].startswith("#")

    def test_supported_formats(self):
        """Test that supported file formats are defined."""
        assert isinstance(SUPPORTED_FORMATS, tuple)
        assert "jpg" in SUPPORTED_FORMATS
        assert "png" in SUPPORTED_FORMATS


class TestConfigurationValues:
    """Test specific configuration value ranges."""

    def test_accuracy_metric_valid(self):
        """Test that accuracy metric is within valid range."""
        accuracy = MODEL_METRICS["accuracy"]
        assert 0.0 <= accuracy <= 1.0

    def test_all_metrics_valid(self):
        """Test that all metrics are in valid ranges."""
        for metric_name, value in MODEL_METRICS.items():
            if isinstance(value, float):
                assert 0.0 <= value <= 1.0 or metric_name.endswith("samples")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
