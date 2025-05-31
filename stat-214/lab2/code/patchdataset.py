from torch.utils.data import Dataset
import torch
import numpy as np


class PatchDataset(Dataset):
    """
    Wrapper dataset class for our datasets of patches
    """

    def __init__(self, patches, transform=None):
        self.patches = patches
        self.transform = transform
        # self.transform is for any torchvision transforms
        # you want to apply to the patches (for data augmentation)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        sample = self.patches[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

class LabeledPatchDataset(Dataset):
    """
    Dataset for (patch, label) pairs used in supervised classification.
    Each patch should be of shape (8, 9, 9).
    """

    def __init__(self, patches, labels, transform=None):
        assert len(patches) == len(labels), "Mismatch between patches and labels"
        self.patches = patches
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x = self.patches[idx]
        y = self.labels[idx]

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        else:
            x = torch.tensor(x, dtype=torch.float32)

        # Label to float tensor (for BCE loss)
        y = float(y)
        y = torch.tensor(y, dtype=torch.float32)

        if self.transform:
            x = self.transform(x)

        return x, y


        