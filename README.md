# ResNet50 For CIFAR-100 And ImageNet2012

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains two standalone PyTorch training scripts built around ResNet-50:

- `resnet50_cifar100.py`: a CIFAR-100 training pipeline with a small-image ResNet-50 variant
- `resnet50_imagenet.py`: an ImageNet-1K training pipeline with a standard ResNet-50 stem and ImageNet-scale augmentation

The focus of this repo is not packaging or framework abstraction. The point is to keep the full training logic visible in a small number of files while still including practical techniques such as mixed precision, stochastic depth, label smoothing, warmup cosine scheduling, checkpoint saving, and TensorBoard logging.

## What Is Included

```text
resnet50-cifar100-imagenet2012/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── resnet50_cifar100.py
└── resnet50_imagenet.py
```

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Script Overview

### 1. `resnet50_cifar100.py`

This script trains a ResNet-50-style model on CIFAR-100.

Key characteristics:

- Uses a CIFAR-style stem: `3x3` convolution, stride `1`, no initial max-pool
- Uses bottleneck residual blocks with stochastic depth
- Uses CIFAR-100 normalization statistics
- Applies CutMix, ColorJitter, RandomHorizontalFlip, and RandomErasing
- Uses label smoothing loss
- Uses SGD with momentum + warmup cosine learning-rate scheduling
- Uses AMP through `autocast` and `GradScaler`
- Saves the best checkpoint and logs to TensorBoard
- Supports multi-GPU `DataParallel`

How to run:

```bash
python resnet50_cifar100.py
```

Dataset behavior:

- CIFAR-100 is downloaded automatically by `torchvision`
- Default data directory: `./data`
- Override with environment variable: `CIFAR100_DATA_DIR`

Output behavior:

- Default checkpoint path: `outputs/checkpoints/resnet50_cifar100_best.pth`
- Default TensorBoard log dir: `outputs/runs/resnet50_cifar100`
- Override with:
  - `RESNET50_CIFAR100_CHECKPOINT`
  - `RESNET50_CIFAR100_LOG_DIR`

### 2. `resnet50_imagenet.py`

This script trains a ResNet-50-style model on ImageNet-1K / ImageNet2012.

Key characteristics:

- Uses the standard ImageNet ResNet stem: `7x7` stride-2 conv + max-pool
- Uses bottleneck residual blocks with stochastic depth
- Applies RandomResizedCrop, ColorJitter, RandomHorizontalFlip, and RandomErasing
- Uses MixUp in the training loop
- Uses label smoothing loss
- Supports both SGD and AdamW through the config
- Uses warmup cosine scheduling and checkpoint resume logic
- Uses AMP and optional multi-GPU `DataParallel`
- Designed for folder-based ImageNet training with `ImageFolder`

How to run:

```bash
python resnet50_imagenet.py
```

Dataset layout expected by default:

```text
data/
└── imagenet/
    ├── train/
    │   ├── n01440764/
    │   ├── n01443537/
    │   └── ...
    └── val/
        ├── n01440764/
        ├── n01443537/
        └── ...
```

You can also override paths with:

- `IMAGENET_TRAIN_DIR`
- `IMAGENET_VAL_DIR`

Output behavior:

- Default checkpoint path: `outputs/checkpoints/resnet50_imagenet_best.pth`
- Default TensorBoard log dir: `outputs/runs/resnet50_imagenet`
- Override with:
  - `RESNET50_IMAGENET_CHECKPOINT`
  - `RESNET50_IMAGENET_LOG_DIR`

## Architecture Notes

Both scripts implement ResNet-50 from scratch using bottleneck blocks, but they are intentionally not identical.

CIFAR-100 variant:

- Adjusted for `32x32` inputs
- Removes the large ImageNet stem and max-pooling front-end
- Keeps feature extraction deeper in the network where it matters more for small images

ImageNet variant:

- Uses the standard large-image stem
- Keeps the ImageNet preprocessing and augmentation recipe closer to conventional ResNet training
- Includes resume logic because long-running training is more common on ImageNet

## Implementation Highlights

### Shared training ideas

Both scripts use several practical training techniques rather than a bare-bones ResNet implementation:

- bottleneck residual blocks
- stochastic depth / drop-path style regularization
- label smoothing
- automatic mixed precision
- warmup cosine learning-rate scheduling
- TensorBoard logging
- checkpoint saving of best validation accuracy

### CIFAR-100-specific choices

The CIFAR-100 script leans into regularization and small-image adaptation:

- CIFAR stem instead of ImageNet stem
- CutMix instead of MixUp
- stronger small-image augmentation stack
- dropout before the classifier head
- early stopping to stop wasted epochs once validation stalls

### ImageNet-specific choices

The ImageNet script is closer to a classic large-scale training setup:

- ImageNet stem and preprocessing
- MixUp inside the training loop
- explicit top-1 and top-5 accuracy tracking
- checkpoint resume support for long training runs
- environment-variable-based dataset paths so the script works cleanly outside the original author machine

## Notes For Users

- This repository does not include pretrained weights or ImageNet data
- CIFAR-100 downloads automatically on first run
- ImageNet must be prepared by the user
- Generated checkpoints, logs, and datasets are ignored by Git via `.gitignore`

## License

This project is released under the MIT License. See `LICENSE` for details.
