import os
import random
import shutil
from glob import glob
from tqdm import tqdm

random.seed(42)

SOURCE_DIR = "/Users/yannisbalasis/Documents/thesis/dataset_binary_class_all"
OUTPUT_DIR = "/Users/yannisbalasis/Documents/thesis/dataset_binary_class_split"

def collect_images():
    data = []

    # Tumor images (glioma, meningioma, pituitary)
    tumor_dir = os.path.join(SOURCE_DIR, 'tumor')
    for subfolder in os.listdir(tumor_dir):
        sub_path = os.path.join(tumor_dir, subfolder)
        if os.path.isdir(sub_path):
            for file in glob(os.path.join(sub_path, '*')):
                data.append((file, 'tumor'))

    # Notumor images
    notumor_dir = os.path.join(SOURCE_DIR, 'no_tumor', 'notumor')
    for file in glob(os.path.join(notumor_dir, '*')):
        data.append((file, 'notumor'))

    return data

def split_data(data):
    random.shuffle(data)
    total = len(data)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)
    return data[:train_end], data[train_end:val_end], data[val_end:]

def copy_images(split_data, split_name):
    for img_path, label in tqdm(split_data, desc=f'Copying {split_name}'):
        filename = os.path.basename(img_path)
        target_dir = os.path.join(OUTPUT_DIR, split_name, label)
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy2(img_path, os.path.join(target_dir, filename))

def main():
    data = collect_images()
    train_data, val_data, test_data = split_data(data)

    print(f"Total images: {len(data)}")
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    copy_images(train_data, 'Training')
    copy_images(val_data, 'Validation')
    copy_images(test_data, 'Testing')

    print(" Dataset split complete!")

if __name__ == '__main__':
    main()
