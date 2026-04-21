"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

# Add app directory to path for test imports
sys.path.insert(0, str(Path(__file__).parent / "app"))


@pytest.fixture
def dummy_image():
    """Create a dummy image for testing."""
    return Image.new("RGB", (256, 256), color="white")


@pytest.fixture
def dummy_grayscale_image():
    """Create a dummy grayscale image for testing."""
    return Image.new("L", (256, 256), color=128)


@pytest.fixture
def dummy_array():
    """Create a dummy numpy array for testing."""
    return np.random.rand(1, 256, 256, 3).astype(np.float32)


@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def app_dir(project_root):
    """Get app directory."""
    return project_root / "app"
