"""
写Dataset的基类
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

PROJECT_PATH = Path(__file__).resolve().parent.parent


class LeavesDataset(Dataset):
    def __init__(self, csv_file, data_root, transform=None, label_to_idx=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.images = self.df["image"].tolist()
        self.data_root = Path(data_root)
        if label_to_idx is None:  # 将 label 转换为索引
            classes = sorted(self.df["label"].unique())
            self.label_to_idx = {name: i for i, name in enumerate(classes)}
        else:
            self.label_to_idx = label_to_idx
        self.targets = self.df["label"].map(self.label_to_idx).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.data_root / self.images[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.targets[idx])


class LeavesTestDataset(Dataset):
    """测试集：只有图片路径，无标签。返回 (image, image_path_str)。"""

    def __init__(self, csv_file, data_root, transform=None):
        self.df = pd.read_csv(csv_file)
        self.images = self.df["image"].tolist()
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.images[idx]
        image = Image.open(self.data_root / rel_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, rel_path


if __name__ == "__main__":
    data_root = PROJECT_PATH / "data" / "classify-leaves"
    tfm = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )
    ds = LeavesDataset(data_root / "train.csv", data_root, transform=tfm)

    # 只划分下标，不存 csv；train/val 共用同一个 ds（同一套 label_to_idx）
    indices = np.arange(len(ds))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=ds.targets,
    )
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # print(len(ds), len(train_ds), len(val_ds), len(ds.label_to_idx))
    # imgs, labels = next(iter(train_loader))
    # print(imgs.shape, labels.shape, labels.dtype)


