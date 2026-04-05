# import cv2
# import os
# import glob
# import numpy as np
#
# # Tạo một dictionary map từ thư mục gốc (ổ D) sang thư mục mới (ổ E)
# # Bác có thể sửa phần '/mnt/e/imagenet_processed/masks/...' theo đúng ý bác muốn
# path_mapping = {
#     '/mnt/d/imagenet/data/masks/train': '/mnt/e/imagenet_processed/masks/train',
#     '/mnt/d/imagenet/data/masks/val': '/mnt/e/imagenet_processed/masks/val',
#     '/mnt/d/imagenet/data/masks/test': '/mnt/e/imagenet_processed/masks/test'
# }
#
# print("Bắt đầu chuyển đổi 255 sang 1 và lưu sang ổ E...")
#
# for in_dir, out_dir in path_mapping.items():
#     # Tự động tạo thư mục đích bên ổ E nếu nó chưa tồn tại
#     os.makedirs(out_dir, exist_ok=True)
#
#     # Tìm tất cả file .png trong thư mục gốc
#     image_paths = glob.glob(os.path.join(in_dir, '*.png'))
#     print(f"Đang xử lý {len(image_paths)} file từ: {in_dir} \n--> Lưu tới: {out_dir}")
#
#     for img_path in image_paths:
#         # Đọc ảnh gốc
#         mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#
#         if mask is not None:
#             # Đổi 255 thành 1
#             mask[mask == 255] = 1
#
#             # Lấy tên file gốc (VD: ISIC_0000000_segmentation.png)
#             filename = os.path.basename(img_path)
#
#             # Ghép tên file vào đường dẫn thư mục mới bên ổ E
#             out_path = os.path.join(out_dir, filename)
#
#             # Lưu ảnh sang ổ E
#             cv2.imwrite(out_path, mask)
#
# print("\n🎉 Đã xong! Toàn bộ mask sạch (0 và 1) đã nằm an toàn bên ổ E.")