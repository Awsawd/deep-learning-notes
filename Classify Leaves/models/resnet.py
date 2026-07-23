import torch
import torch.nn as nn
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
from scripts import dataset_common as dc
import sys

from sklearn.model_selection import train_test_split

from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader, Subset
import pandas as pd


if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


class BasicBlock(nn.Module):
    """残差基本块：out = ReLU(F(x) + shortcut(x))"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 主路第一层：负责可选的降采样（stride=2 时高宽减半）
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 主路第二层：通道不变，不再降采样
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 形状对不上时，用 1×1 卷积把 shortcut 对齐到主路
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=176):
        super().__init__()
        # stem：快速降分辨率
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 组残差层，每组 2 个 BasicBlock → ResNet18
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    DO_TRAIN = True
    NUM_EPOCHS = 10
    PATIENCE = 5
    BATCH_SIZE = 32
    CKPT_PATH = PROJECT_ROOT / "outputs" / "resnet18.pth"
    data_root = PROJECT_ROOT / "data" / "classify-leaves"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (PROJECT_ROOT / "outputs").mkdir(parents=True, exist_ok=True)

    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    ds = dc.LeavesDataset(data_root / "train.csv", data_root, transform = val_tfm)

    if DO_TRAIN:
        indices = np.arange(len(ds))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=ds.targets
        )

        train_base = dc.LeavesDataset(
            data_root / "train.csv", data_root, transform=train_tfm, label_to_idx=ds.label_to_idx
        )
        val_base = dc.LeavesDataset(
            data_root / "train.csv", data_root, transform=val_tfm, label_to_idx=ds.label_to_idx
        )

        train_loader = DataLoader(
            Subset(train_base, train_idx), batch_size=BATCH_SIZE, shuffle=True,
        )
        val_loader = DataLoader(
            Subset(val_base, val_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=4
        )

        model = ResNet18(num_classes=len(ds.label_to_idx)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_val_acc = -1.0
        epochs_no_improve = 0

        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss , correct , total = 0.0, 0, 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            model.eval()
            val_loss , correct , total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    pred = model(images).argmax(1)
                    correct += pred.eq(labels).sum().item()
                    total += labels.size(0)
            val_acc = correct / total
            
            print(
                f"Epoch {epoch+1}, "
                f"train loss: {train_loss/len(train_loader):.4f}, "
                f"val acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "label_to_idx": ds.label_to_idx,
                        "num_classes": len(ds.label_to_idx),
                        "best_val_acc": best_val_acc,
                        "best_epoch": epoch + 1,
                    },
                    CKPT_PATH,
                )
                print("  ↑ saved best")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"early stop (best {best_val_acc:.4f})")
                    break

        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"loaded best epoch {ckpt['best_epoch']}")
    else:
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model = ResNet18(num_classes=ckpt["num_classes"]).to(device)
        model.load_state_dict(ckpt["model"])
        ds.label_to_idx = ckpt["label_to_idx"]

     # ----- test + submission -----
    idx_to_label = {i: n for n, i in ds.label_to_idx.items()}
    test_loader = DataLoader(
        dc.LeavesTestDataset(data_root / "test.csv", data_root, transform=val_tfm),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    model.eval()
    paths, preds = [], []
    with torch.no_grad():
        for imgs, ps in test_loader:
            pred = model(imgs.to(device)).argmax(1).cpu().tolist()
            paths.extend(ps)
            preds.extend(idx_to_label[i] for i in pred)
    out = PROJECT_ROOT / "outputs" / "submission_resnet.csv"
    pd.DataFrame({"image": paths, "label": preds}).to_csv(out, index=False)
    print("wrote", out)
