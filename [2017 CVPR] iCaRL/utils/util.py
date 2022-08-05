import os
import torch
import numpy as np
from sklearn import preprocessing
from avalanche.benchmarks.utils import (
    AvalancheDataset,
    PathsDataset,
    common_paths_root,
)

__all__ = ["icarl_cifar100_augment_data", "get_dataset_per_pixel_mean", "get_image_list", "dataset_list", "PathDataset"]

def icarl_cifar100_augment_data(img):
    img = img.numpy()
    padded = np.pad(img, ((0, 0), (4, 4), (4, 4)), mode="constant")
    random_cropped = np.zeros(img.shape, dtype=np.float32)
    crop = np.random.randint(0, high=8 + 1, size=(2,))

    # Cropping and possible flipping
    if np.random.randint(2) > 0:
        random_cropped[:, :, :] = padded[
            :, crop[0] : (crop[0] + 250), crop[1] : (crop[1] + 250)
        ]
    else:
        random_cropped[:, :, :] = padded[
            :, crop[0] : (crop[0] + 250), crop[1] : (crop[1] + 250)
        ][:, :, ::-1]
    t = torch.tensor(random_cropped)
    
    return t

def get_dataset_per_pixel_mean(dataset, transform):
    result = None
    patterns_count = 0

    for img_path in dataset:
        common_root, exp_paths_list = common_paths_root(img_path)
        paths_dataset = PathsDataset(common_root, exp_paths_list, transform=transform)

        for img_pattern in paths_dataset:
            img_pattern = img_pattern[0]

            if result is None:
                result = torch.zeros_like(img_pattern, dtype=torch.float)

            result += img_pattern
            patterns_count += 1

    if result is None:
        result = torch.empty(0, dtype=torch.float)
    else:
        result = result / patterns_count

    return result

def get_image_list(data_path):
    image = []
    label = []

    data_list = os.listdir(data_path)
    le = preprocessing.LabelEncoder()
    target = le.fit_transform(data_list)

    for idx, folder in enumerate(data_list):
        folder_name = os.path.join(data_path, folder)
        folder_list = os.listdir(folder_name)
        for image_name in folder_list:
            img_name = os.path.join(folder_name, image_name)
            image.append(img_name)
            label.append(target[idx])

    return image, label

def dataset_list(x_data, y_data):
    dataset = []
    file_list = []

    for idx, data in enumerate(x_data):
        instance_tuple = (data, y_data[idx])
        file_list.append(instance_tuple)

    dataset.append(file_list)

    return dataset

def PathDataset(dataset, transfrom, mode):
    for list_of_files in dataset:
        common_root, exp_paths_list = common_paths_root(list_of_files)
        paths_dataset = PathsDataset(common_root, exp_paths_list)
        stream_datasets = AvalancheDataset(paths_dataset, transform_groups=transfrom, initial_transform_group=mode)

    return stream_datasets