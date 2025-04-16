import torch
from torch.utils.data import Dataset
import scipy.io
import numpy as np
import os
from glob import glob

class fUSI(Dataset):
    def __init__(self, folder_path, mat_key='y', transform=None):
        self.mat_files = sorted(glob(os.path.join(folder_path, '*.mat')))
        self.mat_key = mat_key
        self.transform = transform

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        mat_file = self.mat_files[idx]
        data = scipy.io.loadmat(mat_file)
        img = data[self.mat_key].astype(np.float32)  # e.g. key = 'y'

        # Add channel dimension if needed
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)  # (1, H, W)
        elif img.ndim == 3 and img.shape[-1] <= 3:
            img = img.transpose(2, 0, 1)  # (C, H, W)

        img_tensor = torch.tensor(img)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor
