from mmseg.datasets import build_dataset
from mmcv import Config
import numpy as np

# Load config from configuration file
cfg = Config.fromfile('configs/sem_fpn/fpn_lsnet_isic_2018.py')

# Initialize dataset
dataset = build_dataset(cfg.data.train)

# Get first sample
data_item = dataset[0]

# Extract image and label
img = data_item['img'].data
gt_semantic_seg = data_item['gt_semantic_seg'].data

print("--- DATA INSPECTION ---")
print(f"Image shape: {img.shape}")
print(f"Image max value: {img.max()}")
print(f"Image min value: {img.min()}")
print(f"Image mean value: {img.mean()}")

print("-" * 20)
print(f"Unique values in mask (label): {np.unique(gt_semantic_seg)}")