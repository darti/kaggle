from torch.utils.data import Dataset, random_split
import torchvision
import torchvision.transforms as transforms

import pandas as pd
import numpy as np


class Mnist(Dataset):
    def __init__(self, csv_file):
        self.dataset = pd.read_csv(csv_file, dtype=np.float32)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        try:
            idx = index.item()
        except AttributeError:
            idx = index

        img = self.dataset.iloc[idx, 1:].values.reshape(28, 28, 1)
        img = self.transform(img)

        label = self.dataset.iloc[idx, 0]

        return img, label

    def __len__(self):
        return len(self.dataset)


def mnist_dataset(csv_file, eval_percent) -> (Dataset, Dataset):
    data = Mnist(csv_file)
    l = len(data)
    eval_len = int(l * eval_percent)

    ds = random_split(data, [l - eval_len, eval_len])

    return ds[0], ds[1]
