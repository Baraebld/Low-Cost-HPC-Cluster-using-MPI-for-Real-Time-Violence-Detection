# Model Weights Directory

This directory stores the trained deep learning checkpoints.

## Primary Model Checkpoint

- **Filename**: `hybrid_model3.pth`
- **File size**: ~162.7 MB
- **Architecture**: `HybridVideoClassifier` (3D ResNet-18 backbone + Transformer Encoder + Temporal Pooling)
- **Input Dimensions**: `[Batch, 8, 3, 224, 224]` (8 frames sampled uniformly, 224x224 RGB, ImageNet normalized)
- **Framework**: PyTorch (`torch.save(model.state_dict())`)

> [!NOTE]
> Due to GitHub's 100 MB single-file limit, `*.pth` and `*.pt` binary weights are tracked in `.gitignore` rather than committed directly to the repository history.

## Obtaining the Weights

You can obtain the pre-trained weights in either of the following ways:

### Option 1: Automated Download Script
Run the helper script from the repository root:
```bash
python scripts/download_model.py
```

### Option 2: Manual Download
Download `hybrid_model3.pth` directly from the Hugging Face Space repository:
- URL: [https://huggingface.co/spaces/Baraebld/Violence-Detection-model/tree/main/models](https://huggingface.co/spaces/Baraebld/Violence-Detection-model/tree/main/models)
- Place the downloaded file into this `models/` directory:
  ```text
  models/
  ├── hybrid_model3.pth
  ├── model.py
  ├── __init__.py
  └── README.md
  ```
