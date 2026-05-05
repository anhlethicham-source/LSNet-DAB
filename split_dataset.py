import os
from sklearn.model_selection import train_test_split

# Directory paths
img_dir = '/mnt/d/imagenet/data/images/train'
split_dir = '/mnt/d/imagenet/data/splits'

os.makedirs(split_dir, exist_ok=True)

# Get list of image IDs (remove .jpg extension)
all_ids = [f.split('.')[0] for f in os.listdir(img_dir) if f.endswith('.jpg')]

# Split 80% Train, 20% Val (Test)
train_ids, val_ids = train_test_split(all_ids, test_size=0.2, random_state=257)

# Write to files
with open(f'{split_dir}/train_ids.txt', 'w') as f:
    f.write('\n'.join(train_ids))

with open(f'{split_dir}/val_ids.txt', 'w') as f:
    f.write('\n'.join(val_ids))

print(f"Total number of images: {len(all_ids)}")
print(f"Number of training images (80%): {len(train_ids)}")
print(f"Number of validation/test images (20%): {len(val_ids)}")