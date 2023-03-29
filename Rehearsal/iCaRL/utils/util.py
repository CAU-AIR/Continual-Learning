import torch
import numpy as np

__all__ = ["icarl_cifar100_augment_data", "get_dataset_per_pixel_mean"]

def icarl_cifar100_augment_data(img):
    img = img.numpy()
    padded = np.pad(img, ((0, 0), (4, 4), (4, 4)), mode="constant")
    random_cropped = np.zeros(img.shape, dtype=np.float32)
    crop = np.random.randint(0, high=8 + 1, size=(2,))

    # Cropping and possible flipping
    if np.random.randint(2) > 0:
        random_cropped[:, :, :] = padded[
            :, crop[0] : (crop[0] + 32), crop[1] : (crop[1] + 32)
        ]
    else:
        random_cropped[:, :, :] = padded[
            :, crop[0] : (crop[0] + 32), crop[1] : (crop[1] + 32)
        ][:, :, ::-1]
    t = torch.tensor(random_cropped)
    
    return t

def get_dataset_per_pixel_mean(dataset):
    result = None
    patterns_count = 0

    for img_pattern, _ in dataset:
        if result is None:
            result = torch.zeros_like(img_pattern, dtype=torch.float)

        result += img_pattern
        patterns_count += 1

    if result is None:
        result = torch.empty(0, dtype=torch.float)
    else:
        result = result / patterns_count

    return result