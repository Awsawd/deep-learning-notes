# 定义LeNet模型
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

PROJECT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_PATH / "scripts"))
import dataset_common as dc


class LeNet(nn.Module):
    def __init__(self, num_classes=176):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # True: 训练并保存权重；False: 加载已有权重，只做测试集推理
    DO_TRAIN = True
    NUM_EPOCHS = 150
    CKPT_PATH = PROJECT_PATH / "outputs" / "lenet.pth"

    data_root = PROJECT_PATH / "data" / "classify-leaves"
    tfm = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )
    ds = dc.LeavesDataset(data_root / "train.csv", data_root, transform=tfm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (PROJECT_PATH / "outputs").mkdir(parents=True, exist_ok=True)

    if DO_TRAIN:
        indices = np.arange(len(ds))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=42,
            stratify=ds.targets,
        )
        train_loader = DataLoader(
            Subset(ds, train_idx), batch_size=32, shuffle=True
        )
        val_loader = DataLoader(
            Subset(ds, val_idx), batch_size=32, shuffle=False
        )

        model = LeNet(num_classes=len(ds.label_to_idx)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch + 1}, train loss: {train_loss / len(train_loader)}"
                )

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                pred = model(imgs).argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        print("val acc:", correct / total)

        torch.save(
            {
                "model": model.state_dict(),
                "label_to_idx": ds.label_to_idx,
                "num_classes": len(ds.label_to_idx),
            },
            CKPT_PATH,
        )
        print(f"saved weights -> {CKPT_PATH}")
    else:
        if not CKPT_PATH.exists():
            raise FileNotFoundError(
                f"找不到权重文件: {CKPT_PATH}，请先把 DO_TRAIN=True 训练一次"
            )
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model = LeNet(num_classes=ckpt["num_classes"]).to(device)
        model.load_state_dict(ckpt["model"])
        ds.label_to_idx = ckpt["label_to_idx"]
        print(f"loaded weights <- {CKPT_PATH}")

    # --- 测试集推理并写出提交文件 ---
    idx_to_label = {idx: name for name, idx in ds.label_to_idx.items()}
    test_loader = DataLoader(
        dc.LeavesTestDataset(data_root / "test.csv", data_root, transform=tfm),
        batch_size=32,
        shuffle=False,
    )

    model.eval()
    image_paths = []
    pred_labels = []
    with torch.no_grad():
        for imgs, paths in test_loader:
            imgs = imgs.to(device)
            pred = model(imgs).argmax(dim=1).cpu().tolist()
            image_paths.extend(paths)
            pred_labels.extend(idx_to_label[i] for i in pred)

    submission = pd.DataFrame({"image": image_paths, "label": pred_labels})
    out_path = PROJECT_PATH / "outputs" / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"wrote {len(submission)} rows -> {out_path}")
