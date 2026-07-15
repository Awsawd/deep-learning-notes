import dataset_common as dc
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch
import torch.nn as nn

data_root = dc.PROJECT_PATH / "data" / "classify-leaves"
tfm = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
ds = dc.LeavesDataset(data_root / "train.csv", data_root, transform=tfm)
indices = np.arange(len(ds))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=ds.targets)
train_loader = DataLoader(Subset(ds, train_idx), batch_size=32, shuffle=True)
val_loader = DataLoader(Subset(ds, val_idx), batch_size=32, shuffle=False)

num_classes = len(ds.label_to_idx)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3 * 32 * 32, num_classes),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

model.train()
for imgs, labels in train_loader:
    imgs, labels = imgs.to(device), labels.to(device)
    optimizer.zero_grad()
    logits = model(imgs)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
print("train loss:", loss.item())  # 最后一个 batch 的 loss，先够用

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        pred = model(imgs).argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
print("val acc:", correct / total)