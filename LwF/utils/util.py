import os
from sklearn import preprocessing
from avalanche.benchmarks.utils import (
    AvalancheDataset,
    PathsDataset,
    common_paths_root,
)
from avalanche.benchmarks.utils.dataset_definitions import IDataset

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

def PathDataset(dataset, transfrom, mode, task_labels):
    for list_of_files in dataset:
        common_root, exp_paths_list = common_paths_root(list_of_files)
        paths_dataset = PathsDataset(common_root, exp_paths_list)
        stream_datasets = AvalancheDataset(paths_dataset, transform_groups=transfrom, initial_transform_group=mode, task_labels=task_labels)

    return stream_datasets


def PathDataset_v2(dataset, transfrom, mode):
    for list_of_files in dataset:
        common_root, exp_paths_list = common_paths_root(list_of_files)
        paths_dataset = PathsDataset(common_root, exp_paths_list)
        stream_datasets = AvalancheDataset(paths_dataset, transform_groups=transfrom, initial_transform_group=mode)

    return stream_datasets
    
def make_task(num_task, labels):
    start = 0
    max_label = max(labels)
    task_count = max_label // num_task
    task_labels = labels

    for n_task in range(num_task):
        for idx, value in enumerate(labels):
            if value in range(start, task_count*(n_task+1)) and n_task < (num_task-1):
                task_labels[idx] = n_task
            elif n_task == (num_task-1) and value >= start:
                task_labels[idx] = n_task
        start = task_count*(n_task+1)

    return task_labels