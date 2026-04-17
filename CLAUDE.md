# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a medical image segmentation framework based on **LSNet** (Local-to-Global Spatial Kernel Attention Network), built on top of [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). It supports:
- **ISIC 2018** skin lesion segmentation (binary, medical primary task)
- **ADE20K** general scene segmentation (150-class)

## Commands

### Training

```bash
# Single GPU
python tools/train.py configs/sem_fpn/fpn_lsnet_isic_2018.py --work-dir ./work_dirs/myexp

# Multi-GPU distributed (8 GPUs)
./tools/dist_train.sh configs/sem_fpn/fpn_lsnet_isic_2018.py 8 --seed 0 --deterministic

# ADE20K example
./tools/dist_train.sh configs/sem_fpn/fpn_lsnet_b_ade20k_40k.py 8 --seed 0 --deterministic
```

### Evaluation / Testing

```bash
# Single GPU
python tools/test.py configs/sem_fpn/fpn_lsnet_isic_2018.py <checkpoint.pth> --eval mIoU

# Multi-GPU distributed (8 GPUs)
./tools/dist_test.sh configs/sem_fpn/fpn_lsnet_t_ade20k_40k.py pretrain/lsnet_t_semfpn.pth 8 --eval mIoU

# Using eval.sh shortcut
PORT=12345 ./tools/dist_test.sh configs/sem_fpn/fpn_lsnet_t_ade20k_40k.py pretrain/lsnet_t_semfpn.pth 8 --eval mIoU
```

### Dependencies

```bash
pip install mmsegmentation==0.30.0
pip install triton   # For SKA Triton kernels
```

## Architecture

### Model Pipeline

```
Input Image
    ↓
LSNet Backbone  (model/lsnet.py)      — 4-stage hybrid CNN-Attention
    ↓
LSNetFPN Neck   (model/lsnet_fpn.py)  — Feature Pyramid Network
    ↓
FPNHead Decoder (mmsegmentation built-in)
    ↓
Loss: FocalDiceLoss (ISIC 2018) or CrossEntropyLoss (ADE20K)
    ↓
Segmentation Map
```

### Key Source Files

| File | Role |
|------|------|
| `model/lsnet.py` | LSNet backbone — 4 stages, each combining LKP (7×7 large-kernel path) + SKA attention |
| `model/ska.py` | Triton-accelerated Spatial Kernel Attention kernels (forward + backward) |
| `model/lsnet_fpn.py` | FPN neck with lateral convolutions and top-down feature fusion |
| `mmseg_custom/datasets/isic_2018.py` | ISIC 2018 dataset class with medical metrics (Dice, Sensitivity, Specificity, F1) |
| `mmseg_custom/model/loss_function/dice_function.py` | `FocalDiceLoss` and `MyBCEDiceLoss` for binary medical segmentation |
| `mmcv_custom/checkpoint.py` | Modified checkpoint loading (handles partial/mismatched weights) |

### LSNet Backbone Variants

| Variant | `embed_dim` | `depth` | ADE20K mIoU |
|---------|-------------|---------|-------------|
| `lsnet_t` (Tiny) | [64, 128, 256, 384] | [0, 2, 8, 10] | 40.1% |
| `lsnet_s` (Small) | [96, 192, 320, 448] | [1, 2, 8, 10] | 41.6% |
| `lsnet_b` (Base) | [128, 256, 384, 512] | [4, 6, 8, 10] | 43.1% |

Each stage block (`model/lsnet.py:Block`) stacks: **RepVGGDW → LKP (large kernel path) → SKA → FFN**.

### Config System

Configs are MMSegmentation-style Python dicts with inheritance via `_base_`. Main config directories:

- `configs/_base_/datasets/` — `isic_2018.py` (256×256, binary) and `ade20k.py` (512×512, 150-class)
- `configs/_base_/schedules/` — Training schedules (40k, 80k, 160k iterations)
- `configs/sem_fpn/` — Full experiment configs combining model + dataset + schedule

### Medical Segmentation Details

- **Loss**: `FocalDiceLoss` (focal_weight=1.0, dice_weight=0.5) — handles class imbalance in lesion datasets
- **Augmentation**: Horizontal/vertical flip, random rotation up to 180° (configured in `configs/_base_/datasets/isic_2018.py`)
- **Metrics**: Dice, IoU, Recall, Sensitivity, Specificity, Precision, F1 (computed in `mmseg_custom/datasets/isic_2018.py`)
- **Data paths**: Default config points to `/mnt/d/imagenet/` — update `configs/_base_/datasets/isic_2018.py` for your local setup

### Custom Registrations

`mmseg_custom/` registers custom components into MMSegmentation's registry at import time. The dataset (`ISIC2018Dataset`) and loss (`FocalDiceLoss`, `MyBCEDiceLoss`) must be imported before MMSeg builds the pipeline — this happens via the config's `custom_imports` field.
