import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from PIL import Image

import pandas as pd
import numpy as np


class Sample(Dataset):
    def __init__(self, dataDir):
        self.dataDir = dataDir
        self.images = [os.path.join(self.dataDir, f) for f  in os.listdir(self.dataDir)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.images[idx]

        return images


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, inputWidth: int, inputHeight: int) -> None:
        self.dataframe = dataframe
        self.transformEntry = transforms.Compose([
            transforms.Resize((inputHeight, inputWidth), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.transformLabel = transforms.Compose([
            transforms.Resize((inputHeight, inputWidth), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485), std=(0.229))
        ])

    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, index: int):
        entryPath = self.dataframe.iloc[index, 0]
        targetPath = self.dataframe.iloc[index, 1]

        entry = Image.fromarray(np.load(entryPath).astype(np.uint8))
        target = Image.fromarray((np.load(targetPath) * 255.0).astype(np.uint8))

        entry = self.transformEntry(entry)
        target = self.transformLabel(target)

        return (entry, target)