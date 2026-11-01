# tests/test_video_utils.py
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from src.video_utils import load_video_frames


class TestVideoUtils:
    """Test cases for video processing utilities."""

    @patch('src.video_utils.cv2.VideoCapture')
    def test_load_video_frames_success(self, mock_video_capture, sample_config):
        """Test successful video frame loading."""
        # Mock video capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30  # 30 frames total
        mock_cap.read.return_value = (True, np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        frames = load_video_frames("fake_video.mp4")

        assert isinstance(frames, torch.Tensor)
        assert frames.shape[0] == sample_config["NUM_FRAMES"]  # NUM_FRAMES
        assert frames.shape[1] == 3  # Channels
        assert frames.shape[2] == sample_config["FRAME_SIZE"][1]  # Height
        assert frames.shape[3] == sample_config["FRAME_SIZE"][0]  # Width

    @patch('src.video_utils.cv2.VideoCapture')
    def test_load_video_frames_video_not_opened(self, mock_video_capture):
        """Test handling when video cannot be opened."""
        # Mock video capture that fails to open
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        frames = load_video_frames("nonexistent_video.mp4")

        # Should return zeros tensor
        assert isinstance(frames, torch.Tensor)
        assert torch.all(frames == 0)

    @patch('src.video_utils.cv2.VideoCapture')
    def test_load_video_frames_no_frames(self, mock_video_capture):
        """Test handling when video has no frames."""
        # Mock video capture with 0 frames
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0  # No frames
        mock_video_capture.return_value = mock_cap

        frames = load_video_frames("empty_video.mp4")

        # Should return zeros tensor
        assert isinstance(frames, torch.Tensor)
        assert torch.all(frames == 0)