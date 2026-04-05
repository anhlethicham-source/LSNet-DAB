from mmseg.datasets import build_dataset
from mmcv import Config
import numpy as np

# Load config từ file của bạn
cfg = Config.fromfile('configs/sem_fpn/fpn_lsnet_isic_2018.py')

# Khởi tạo dataset
dataset = build_dataset(cfg.data.train)

# Lấy mẫu dữ liệu đầu tiên
data_item = dataset[0]

# Trích xuất ảnh và label
img = data_item['img'].data
gt_semantic_seg = data_item['gt_semantic_seg'].data

print("--- KIỂM TRA DỮ LIỆU ---")
print(f"Shape của ảnh: {img.shape}")
print(f"Giá trị Max của ảnh: {img.max()}")
print(f"Giá trị Min của ảnh: {img.min()}")
print(f"Giá trị Mean của ảnh: {img.mean()}")

print("-" * 20)
print(f"Giá trị duy nhất trong Mask (nhãn): {np.unique(gt_semantic_seg)}")