# tests/conftest.py
"""Pytest configuration and fixtures."""
import pytest
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "NUM_FRAMES": 8,
        "FRAME_SIZE": (224, 224),
        "IMAGENET_MEAN": [0.485, 0.456, 0.406],
        "IMAGENET_STD": [0.229, 0.224, 0.225]
    }