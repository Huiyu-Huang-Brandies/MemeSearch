import os
import shutil
import random

# Original folder containing all meme images
source_folder = "./memes"
# Target folders for train and test splits
train_folder = "./train"
test_folder = "./test"

# Create train and test directories if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Retrieve all image files from the source folder
all_files = [f for f in os.listdir(source_folder) if f.endswith(".png")]
# Shuffle for random splitting
random.seed(42)  # Ensures reproducibility
random.shuffle(all_files)

# Define the split ratio for train and test sets
split_ratio = 0.8
split_idx = int(len(all_files) * split_ratio)

# Split the images into train and test lists
train_files = all_files[:split_idx]
test_files = all_files[split_idx:]

# Copy files to the train folder
for filename in train_files:
    src_path = os.path.join(source_folder, filename)
    dst_path = os.path.join(train_folder, filename)
    shutil.copy(src_path, dst_path)

# Copy files to the test folder
for filename in test_files:
    src_path = os.path.join(source_folder, filename)
    dst_path = os.path.join(test_folder, filename)
    shutil.copy(src_path, dst_path)

print(
    f"Images have been successfully split into {len(train_files)} train and {len(test_files)} test images."
)
