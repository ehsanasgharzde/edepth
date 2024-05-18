from edepth import edepth

from typing import Any

import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split as trainTestSplit

import pandas as pd
import numpy as np



class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, inputWidth: int, inputHeight: int) -> None:
        self.dataframe = dataframe
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        self.transform = transforms.Compose([
            transforms.Resize((self.inputHeight, self.inputWidth), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, index: int) -> Any:
        entryPath = self.dataframe.iloc[index, 0]
        targetPath = self.dataframe.iloc[index, 1]

        entry = np.load(entryPath)
        target = np.load(targetPath)

        if entry.ndim == 4:
            entry = entry[0]
        if entry.ndim == 3 and entry.shape[-1] == 3:
            entry = Image.fromarray(entry.astype(np.uint8))

        if target.ndim == 4:
            target = target[0]
        if target.ndim == 3 and target.shape[-1] == 3:
            target = Image.fromarray(target.astype(np.uint8))
        elif target.ndim == 2:
            target = np.expand_dims(target, axis=2)
            target = Image.fromarray(np.repeat(target, 3, axis=2).astype(np.uint8))

        entry = self.transform(entry)
        target = self.transform(target)

        print(target.shape)

        return entry, target
    
dataDir = r"/home/ehsanasgharzde/Desktop/Projects/depth/dataset.csv"
dataframe = pd.read_csv(dataDir)

train, validate = trainTestSplit(dataframe, test_size=0.2, random_state=42)

trainset = Dataset(train, 224, 224)
validationset = Dataset(validate, 224, 224)

model = edepth()

if __name__ == "__main__":
    model.etrain(trainset, validationset, 50, batchSize=4)
