# SpikeMixer

PyTorch implementation of **SpikeMixer: Dynamic axial mixing for spiking neural networks**.

This project provides the official codebase for training and evaluating SpikeMixer on multiple datasets including Tiny-ImageNet, CIFAR-10, and CIFAR-10DVS.

## Paper

> **SpikeMixer: Dynamic axial mixing for spiking neural networks**
> Neurocomputing, 2025
> [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231226015183)

## Overview

SpikeMixer is a spiking neural network (SNN) architecture that incorporates dynamic token mixing mechanisms for efficient event-based and static image recognition. The model combines:

- **Dynamic Mixing**: Adaptively mixes spatial tokens using multi-head operations
- **Axial Mixing**: Efficient H/W axis processing for 2D spatial data
- **Spiking Neurons**: LIF (Leaky Integrate-and-Fire) neurons for event-based processing
- **Progressive Patch Embedding**: Multi-stage downsampling with spike-based encoding

## Project Structure

```
SpikeMixer/
├── tinyimagenet/         # Tiny-ImageNet training scripts (200 classes)
│   ├── train.py          # Training script
│   ├── test.py           # Evaluation script
│   ├── model.py          # SpikeMixer model for Tiny-ImageNet
│   └── tinyimagenet.yml  # Configuration file
├── cifar10/              # CIFAR-10 training scripts
│   ├── train.py          # Training script
│   ├── model.py          # SpikeMixer model for CIFAR-10
│   ├── loader.py         # Data loader utilities
│   └── cifar10.yml       # Configuration file
├── cifar10dvs/           # CIFAR-10DVS training scripts
│   ├── train.py          # Training script
│   ├── model.py          # SpikeMixer model for DVS data
│   ├── autoaugment.py    # Data augmentation
│   └── utils.py          # Utility functions
└── README.md
```

## Requirements

```bash
# Core dependencies
torch>=1.8.0
torchvision
timm>=0.4.0
spikingjelly>=0.0.9
cupy  # Required for spikingjelly CUDA backend

# Training utilities
tensorboard
wandb
pyyaml

# Optional
apex  # For mixed precision training
```

Install via pip:

```bash
pip install torch torchvision timm spikingjelly tensorboard pyyaml cupy
```

## Dataset Preparation

### Tiny-ImageNet

Download Tiny-ImageNet-200 from http://cs231n.stanford.edu/tiny-imagenet-200.zip. Extract to:

```
datasets/
└── tiny-imagenet-200/
    ├── train/
    ├── val/
    └── test/
```

### ImageNet (Full)

Download and extract ImageNet to a directory:

```
datasets/
└── ImageNet2012/
    ├── train/
    └── validation/
```

### CIFAR-10

```bash
# Will be automatically downloaded on first run
python cifar10/train.py
```

Or download manually to:

```
datasets/
└── cifar-10-python/
```

### CIFAR-10DVS

```bash
# Download CIFAR-10DVS dataset
datasets/
└── CIFAR10DVS/
    ├── CIFAR10DVS/
    │   ├── train/
    │   └── test/
```

## Usage

### Tiny-ImageNet Training

```bash
cd tinyimagenet

# Basic training
python train.py --data-dir ./datasets/tiny-imagenet-200/

# With custom configuration
python train.py -c tinyimagenet.yml --data-dir ./datasets/tiny-imagenet-200/
```

### Tiny-ImageNet Evaluation

```bash
python test.py --data-dir ./datasets/tiny-imagenet-200/ --resume ./pretrained/spikemixer-tinyimagenet-checkpoint.pth.tar
```

### CIFAR-10 Training

```bash
cd cifar10

# Basic training
python train.py --data-dir ./datasets/cifar-10-python/

# With custom configuration
python train.py -c cifar10.yml
```

### CIFAR-10DVS Training

```bash
cd cifar10dvs

# Basic training
python train.py --data-path ./datasets/CIFAR10DVS --T 16

# With mixed precision
python train.py --data-path ./datasets/CIFAR10DVS --T 16 --amp
```

## Citation

If you find this code useful for your research, please cite:

```bibtex
@article{spikemixer,
title = {SpikeMixer: Dynamic axial mixing for spiking neural networks},
journal = {Neurocomputing},
volume = {696},
pages = {134120},
year = {2026},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2026.134120},
url = {https://www.sciencedirect.com/science/article/pii/S0925231226015183},
author = {Jiemin Ji and Liqiang He and Jun Li},
keywords = {Spiking neural network, MLP-mixer},
abstract = {Spiking neural networks (SNNs) have emerged as a promising computational architecture due to their brain-inspired and energy-efficient nature. Although attention mechanisms offer a pathway to improved performance in SNNs, the computational cost associated with their quadratic complexity may offset the energy savings. In this paper, we introduce SpikeMixer, a novel and efficient alternative to conventional attention in SNNs. SpikeMixer leverages the Multi-Layer Perceptron (MLP)-Mixer architecture to efficiently capture long-range dependencies within spiking neural networks. In particular, SpikeMixer uses dynamic mixing and axial mixing, providing content-adaptive and complementary features in two orthogonal axial directions. Extensive evaluations across several static and neuromorphic benchmark datasets validate the efficiency and efficacy of our approach.}
}
```

This implementation builds upon:
- [SpikingJelly](https://github.com/fangwei123456/spikingjelly) for spiking neuron simulation
- [timm](https://github.com/rwightman/pytorch-image-models) for training utilities and model registry
- [PyTorch](https://pytorch.org/) for deep learning framework
