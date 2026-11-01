# tests/test_model_loading.py
import pytest
import torch
from unittest.mock import patch, MagicMock
from src.load_model import load_trained_model, get_device, get_model_info


class TestModelLoading:
    """Test cases for model loading functionality."""

    def test_get_device_cpu(self):
        """Test device detection when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            device = get_device()
            assert device == "cpu"

    def test_get_device_cuda(self):
        """Test device detection when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            device = get_device()
            assert device == "cuda"

    def test_get_device_mps(self):
        """Test device detection when MPS is available."""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=True):
            device = get_device()
            assert device == "mps"

    @patch('src.load_model.Path')
    @patch('src.load_model.torch.load')
    @patch('src.load_model.HybridVideoClassifier')
    def test_load_trained_model_success(self, mock_model_class, mock_torch_load, mock_path):
        """Test successful model loading."""
        # Mock the model class
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Mock torch.load
        mock_torch_load.return_value = {"state_dict": "mock"}

        # Mock Path
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        with patch('torch.cuda.is_available', return_value=False):
            model, device = load_trained_model()

            assert model is not None
            assert device == "cpu"
            mock_model.load_state_dict.assert_called_once()
            mock_model.eval.assert_called_once()

    def test_get_model_info(self):
        """Test model information extraction."""
        # Create a mock model with parameters
        mock_model = MagicMock()
        mock_param1 = MagicMock()
        mock_param1.numel.return_value = 100
        mock_param1.requires_grad = True

        mock_param2 = MagicMock()
        mock_param2.numel.return_value = 50
        mock_param2.requires_grad = False

        mock_model.parameters.return_value = [mock_param1, mock_param2]
        mock_model.__class__.__name__ = "TestModel"

        info = get_model_info(mock_model)

        assert info["total_parameters"] == 150
        assert info["trainable_parameters"] == 100
        assert info["model_type"] == "TestModel"

    def test_load_trained_model_file_not_found(self, tmp_path):
        """Test model loading when file is not found."""
        with pytest.raises(FileNotFoundError):
            load_trained_model(model_path="nonexistent.pth", base_dir=tmp_path)
