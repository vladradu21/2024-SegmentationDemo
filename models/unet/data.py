import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Read image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image / 255.0  # normalize
        image = np.transpose(image, (2, 0, 1))  # transpose (3, 1024, 1024)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Read mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0  # normalize
        mask = np.expand_dims(mask, axis=0)  # (1, 1024, 1024)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples
