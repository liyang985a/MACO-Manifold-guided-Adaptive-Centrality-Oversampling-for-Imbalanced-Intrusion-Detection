# ========== Standard Library ==========
import gc
import json
import math
import os

# ========== Third-Party Libraries ==========
import numpy as np
import pandas as pd
from PIL import Image


def load_and_preprocess_data(data_path, label_column='label', index_col=None):
    # Added index_col parameter to handle the index column
    data = pd.read_csv(data_path, index_col=index_col, low_memory=False)

    numeric_columns = data.columns.tolist()
    numeric_columns.remove(label_column)

    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
    data[numeric_columns] = data[numeric_columns].fillna(0)

    return data, numeric_columns


def determine_image_size(num_features):
    size = math.ceil(math.sqrt(num_features))
    return size, size


def normalize_features(features):
    min_val = features.min()
    max_val = features.max()
    if min_val == max_val:
        return np.zeros_like(features, dtype=np.uint8)
    normalized = (features - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)


def process_batch(data, start_idx, end_idx, numeric_columns, image_size, output_folder, prefix, label_dict):
    height, width = image_size
    for i in range(start_idx, end_idx):
        sample = data.iloc[i]
        image_name = f'{prefix}_{i}.png'

        features = sample[numeric_columns].values.astype(float)
        normalized = normalize_features(features)

        total_pixels = height * width
        if len(normalized) < total_pixels:
            normalized = np.pad(normalized, (0, total_pixels - len(normalized)), 'constant')
        else:
            normalized = normalized[:total_pixels]

        image_array = normalized.reshape((height, width)).astype(np.uint8)

        image_path = os.path.join(output_folder, image_name)
        image = Image.fromarray(image_array)
        image.save(image_path)

        label_dict[image_name] = int(sample['label'])

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} images")
            gc.collect()


def process_dataset(data, numeric_columns, image_size, output_folder, prefix):
    os.makedirs(output_folder, exist_ok=True)
    label_dict = {}

    batch_size = 10000
    num_samples = len(data)
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        process_batch(data, batch_start, batch_end, numeric_columns, image_size, output_folder, prefix, label_dict)

    return label_dict


def main(output_base_dir):
    # Path configuration
    train_data_path = 'final_multiclass_augmented3.csv'
    test_data_path = 'multi_test.csv'

    train_output_folder = os.path.join(output_base_dir, 'output_train_images_3')
    test_output_folder = os.path.join(output_base_dir, 'output_test_images_3')
    train_json_path = os.path.join(output_base_dir, 'image_label_dict_train_3.json')
    test_json_path = os.path.join(output_base_dir, 'image_label_dict_test_3.json')

    # Load data (Key modification)
    # Training set does not ignore the index column (since it has none in this augmented version)
    train_data, numeric_columns = load_and_preprocess_data(train_data_path, index_col=None)

    # Test set needs to ignore the first column index
    test_data, _ = load_and_preprocess_data(test_data_path, index_col=0)

    # Force sync test set column names (Core solution to mismatched features)
    test_data.columns = train_data.columns

    image_size = determine_image_size(len(numeric_columns))
    print(f"Image size set to: {image_size[0]}x{image_size[1]}")

    # Process training set
    print("\nProcessing training set...")
    train_label_dict = process_dataset(train_data, numeric_columns, image_size, train_output_folder, 'train')
    with open(train_json_path, 'w') as f:
        json.dump(train_label_dict, f)

    # Process test set (column names are now aligned)
    print("\nProcessing test set...")
    test_label_dict = process_dataset(test_data, numeric_columns, image_size, test_output_folder, 'test')
    with open(test_json_path, 'w') as f:
        json.dump(test_label_dict, f)

    print("\nAll images and label dictionaries processed successfully!")


if __name__ == "__main__":
    # 👇 Specify your desired save path here
    output_base_dir = 'D:/data/unsw-nb15over'
    main(output_base_dir)