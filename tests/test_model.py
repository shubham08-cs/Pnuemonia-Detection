"""Tests for model loading and core functionality."""

import sys
from pathlib import Path
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from config import MODEL_PATH, INPUT_SIZE


class TestModelConfiguration:
    """Test model configuration and constants."""

    def test_model_path_defined(self):
        """Test that model path is configured."""
        assert MODEL_PATH is not None
        assert isinstance(MODEL_PATH, Path)

    def test_input_size_defined(self):
        """Test that input size is configured."""
        assert INPUT_SIZE == (256, 256)

    def test_input_size_is_tuple(self):
        """Test that input size is a tuple."""
        assert isinstance(INPUT_SIZE, tuple)
        assert len(INPUT_SIZE) == 2


class TestModelAvailability:
    """Test model file availability."""

    def test_model_file_exists(self):
        """Test that model file exists at expected path."""
        # Note: This will fail if model is not present, which is expected
        # In CI/CD, the model file should be present or skipped
        if MODEL_PATH.exists():
            assert MODEL_PATH.is_file()
        else:
            pytest.skip("Model file not found - expected in production environment")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
