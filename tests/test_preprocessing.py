"""Tests for image preprocessing functions."""

import numpy as np
import pytest
from PIL import Image

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from helpers import preprocess_image


class TestImagePreprocessing:
    """Test image preprocessing pipeline."""

    def test_preprocess_image_shape(self):
        """Test that preprocessed image has correct shape."""
        # Create a dummy image
        dummy_img = Image.new("RGB", (512, 512), color="white")
        
        # Preprocess
        result = preprocess_image(dummy_img, target_size=(256, 256))
        
        # Check shape: (1, 256, 256, 3)
        assert result.shape == (1, 256, 256, 3), f"Expected shape (1, 256, 256, 3), got {result.shape}"

    def test_preprocess_image_values_normalized(self):
        """Test that image values are normalized to [0, 1]."""
        dummy_img = Image.new("RGB", (512, 512), color="white")
        result = preprocess_image(dummy_img)
        
        assert np.min(result) >= 0.0, "Image has values below 0"
        assert np.max(result) <= 1.0, "Image has values above 1"

    def test_preprocess_grayscale_image(self):
        """Test preprocessing of grayscale image."""
        dummy_img = Image.new("L", (512, 512), color=128)
        result = preprocess_image(dummy_img)
        
        # Should convert to RGB, so shape is (1, 256, 256, 3)
        assert result.shape == (1, 256, 256, 3)

    def test_preprocess_image_batching(self):
        """Test that single image is batched correctly."""
        dummy_img = Image.new("RGB", (128, 128), color="black")
        result = preprocess_image(dummy_img)
        
        # First dimension should be batch size = 1
        assert result.shape[0] == 1


class TestImageInputValidation:
    """Test input validation for images."""

    def test_invalid_image_type(self):
        """Test handling of invalid image types."""
        with pytest.raises((AttributeError, TypeError)):
            preprocess_image("not_an_image")

    def test_empty_image(self):
        """Test handling of empty/null images."""
        with pytest.raises((AttributeError, TypeError)):
            preprocess_image(None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
