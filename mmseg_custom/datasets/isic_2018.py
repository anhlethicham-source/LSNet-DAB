from mmseg.datasets import CustomDataset, DATASETS
import numpy as np


@DATASETS.register_module(force=True)
class ISIC2018Dataset(CustomDataset):
    # Định nghĩa tên lớp
    CLASSES = ('background', 'lesion')

    # Định nghĩa màu sắc khi xem ảnh kết quả (R, G, B)
    PALETTE = [[0, 0, 0], [255, 0, 0]]

    def __init__(self, **kwargs):
        # img_suffix: đuôi file ảnh gốc (.jpg)
        # seg_map_suffix: đuôi file ảnh mask (.png)
        super().__init__(img_suffix='.jpg', seg_map_suffix='_segmentation.png', **kwargs)

