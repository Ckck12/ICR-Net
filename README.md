# Are Deepfake Detectors Robust to Temporal Corruption?

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

ICR-Net is a robust deepfake detection method designed to handle temporal corruptions in video streams. The method estimates frame reliability and selectively restores corrupted frames through integrity-aware contrastive learning on clean-corrupted pairs, learning corruption-invariant and class-separable features.

### Key Features

- **Temporal Integrity Assessment**: GRU-based frame reliability estimation
- **Selective Frame Correction**: Adaptive residual correction based on integrity scores
- **Contrastive Learning**: Corruption-invariant feature learning from clean/corrupted pairs
- **Robust Classification**: Frame-level classification with temporal consistency
- **Cross-dataset Generalization**: Strong performance across different datasets under temporal corruptions

## Architecture

```
Input Video (Clean/Corrupt Pair)
    ↓
Spatial Encoder (ResNet34)
    ↓
Integrity Assessment (GRU)
    ↓
Residual Prediction (1D-CNN)
    ↓
Selective Correction
    ↓
Contrastive Learning
    ↓
Frame Classification
    ↓
Video-level Prediction
```

## Quick Start

### Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd ICR-Net

# Create and activate virtual environment
conda create -n icr-net python=3.9
conda activate icr-net

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

1. **Clean Data**: FaceForensics++ clean videos
2. **Corrupt Data**: Videos with various corruptions applied
   - Supported corruptions: `bit_error`, `h264_crf`, `h264_abr`, `h265_crf`, `h265_abr`, `motion_blur`, `packet_loss`

### Training

```bash
# Train with single corruption type
python scripts/train.py \
    --config src/configs/icr_net.yaml \
    --train_corruption packet_loss \
    --train_severity 3 \
    --output_dir ./checkpoints

# Distributed training
bash scripts/train_distributed.sh
```

### Inference

```bash
# Single video test
python scripts/test.py \
    --config src/configs/icr_net.yaml \
    --weights ./checkpoints/best_model.pth \
    --input_video ./test_video.mp4

# Batch test
python scripts/test_batch.py \
    --config src/configs/icr_net.yaml \
    --weights ./checkpoints/best_model.pth \
    --test_corruption packet_loss \
    --test_severity 3
```

## Project Structure

```
ICR-Net/
├── src/
│   ├── models/
│   │   └── icr_net.py          # ICR-Net main model
│   ├── datasets/
│   │   └── pair_dataset.py     # Clean/Corrupt pair dataset
│   ├── utils/
│   │   └── metrics.py          # Evaluation metrics
│   └── configs/
│       └── icr_net.yaml        # Model configuration
├── scripts/
│   ├── train.py                # Training script
│   ├── test.py                 # Testing script
│   ├── train_distributed.sh    # Distributed training
│   └── test_batch.py           # Batch testing
├── examples/
│   ├── train_example.py        # Training example
│   └── inference_example.py    # Inference example
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Configuration

### Model Configuration (src/configs/icr_net.yaml)

```yaml
# Model settings
model_name: icr_net
video_mode: true
clip_size: 16
feature_dim: 512
gru_hidden_dim: 512
proj_dim: 256

# Loss weights
lambda_cls: 1.0        # Classification loss
lambda_pred: 1.0       # Prediction loss
lambda_con: 0.5       # Contrastive loss
lambda_sc: 0.01       # Selective correction regularization

# Training settings
train_batchSize: 8
nEpochs: 50
optimizer:
  type: adam
  adam:
    lr: 0.0001
```

## Development

### Code Structure

- **ICR-Net Model**: `src/models/icr_net.py`
- **Data Loader**: `src/datasets/pair_dataset.py`
- **Loss Functions**: Integrated in model
- **Training Script**: `scripts/train.py`

### Testing

```bash
# Run examples
python examples/train_example.py
python examples/inference_example.py

# Test model
python scripts/test.py --config src/configs/icr_net.yaml --weights model.pth --input_video test.mp4
```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**ICR-Net** - Robust Deepfake Detection through Integrity-aware Contrastive Learning
