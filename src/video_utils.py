# src/video_utils.py
import cv2
import numpy as np
import torch
import logging
from typing import Optional, Tuple
from configs.config import NUM_FRAMES, FRAME_SIZE, IMAGENET_MEAN, IMAGENET_STD

# Setup logging
logger = logging.getLogger(__name__)

def load_video_frames(
    video_path: str,
    num_frames: int = NUM_FRAMES,
    resize: Tuple[int, int] = FRAME_SIZE
) -> torch.Tensor:
   
    frames = []
    last_frame = np.zeros((resize[1], resize[0], 3), dtype=np.uint8)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Cannot open video: {video_path}")
        return torch.zeros((num_frames, 3, resize[1], resize[0]), dtype=torch.float)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"⚠️ Video has 0 frames: {video_path}")
        return torch.zeros((num_frames, 3, resize[1], resize[0]), dtype=torch.float)

    frame_idxs = np.linspace(0, max(total_frames-1,0), num_frames).astype(int)

    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            frame = last_frame
        else:
            try:
                # If grayscale, convert to RGB
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                # If frame has wrong number of channels
                elif frame.shape[2] != 3:
                    frame = last_frame
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, resize)
            except:
                frame = last_frame

        last_frame = frame
        frame = frame.astype(np.float32)/255.0
        frame = (frame - IMAGENET_MEAN)/IMAGENET_STD
        frame = np.transpose(frame, (2,0,1))
        frames.append(frame)

    cap.release()
    return torch.tensor(np.array(frames), dtype=torch.float)

