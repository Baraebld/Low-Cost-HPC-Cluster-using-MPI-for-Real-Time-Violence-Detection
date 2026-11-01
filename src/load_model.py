# src/load_model.py
import torch
import os
import logging
from pathlib import Path
from typing import Tuple, Optional
from models.model import HybridVideoClassifier
from configs.config import NUM_FRAMES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_device() -> str:
    """Get the best available device for inference."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using MPS (Apple Silicon) device")
    else:
        device = "cpu"
        logger.info("Using CPU device")
    return device

def load_trained_model(
    model_path: str = "models/hybrid_model3.pth",
    base_dir: Optional[str] = None
) -> Tuple[torch.nn.Module, str]:
    """
    Load the trained violence detection model.

    Args:
        model_path: Path to the model weights file
        base_dir: Base directory for the project (auto-detected if None)

    Returns:
        Tuple of (loaded model, device string)
    """
    try:
        # Determine base directory
        if base_dir is None:
            # Try to find project root
            current_file = Path(__file__).resolve()
            base_dir = current_file.parent.parent

        # Convert to Path object
        base_path = Path(base_dir)

        # Get device
        device = get_device()

        # Initialize model
        logger.info(f"Initializing model with {NUM_FRAMES} frames per video")
        model = HybridVideoClassifier(
            num_frames=NUM_FRAMES,
            num_classes=1,
            freeze_backbone=True
        ).to(device)

        # Load model weights
        model_file_path = base_path / model_path
        if not model_file_path.exists():
            # Try alternative locations
            alt_paths = [
                Path(model_path),  # Absolute path
                base_path / "models" / "hybrid_model3.pth",  # Default name
                Path.cwd() / model_path,  # Current working directory
            ]

            for alt_path in alt_paths:
                if alt_path.exists():
                    model_file_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Model file not found. Searched in: {[str(p) for p in alt_paths]}")

        logger.info(f"Loading model weights from: {model_file_path}")
        state_dict = torch.load(model_file_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        # Set to evaluation mode
        model.eval()

        logger.info("Model loaded successfully")
        return model, device

    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}") from e

def get_model_info(model: torch.nn.Module) -> dict:
    """Get information about the loaded model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_type": type(model).__name__,
    }