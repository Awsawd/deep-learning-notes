# deep-learning-notes

个人深度学习 / 机器学习学习笔记仓库。每个子目录是一个独立学习主题。

## 目录结构

```
deep-learning-notes/
├── california-house-prices/   # 加州房价预测（EDA → RF → PyTorch MLP）
├── Classify Leaves/           # 叶片分类（LeNet → AlexNet → VGG → ResNet）
└── README.md                  # 本文件（主题说明集中在此）
```

## 当前主题

| 目录 | 内容 | 状态 |
|------|------|------|
| [california-house-prices](./california-house-prices/) | 加州房价：Random Forest 基线 + PyTorch MLP | 进行中 |
| [Classify Leaves](./Classify%20Leaves/) | 叶片图像分类：经典 CNN 架构进阶 | 进行中（阶段 3 VGG） |

## 环境

可复用 conda 环境 `california-dl`（需 `torch`、`torchvision`、`pandas`、`Pillow`、`scikit-learn` 等）。

```bash
conda activate california-dl
```

---

# Classify Leaves — CNN 架构学习

用 [Kaggle Classify Leaves](https://www.kaggle.com/competitions/classify-leaves) 风格数据，按历史顺序亲手实现并理解：

**LeNet → AlexNet → VGG → ResNet**

目标不是刷榜，而是搞清楚：卷积怎么提特征、网络为什么越来越深、残差解决了什么问题。

## 数据概况

| 项目 | 内容 |
|------|------|
| 路径 | `Classify Leaves/data/classify-leaves/` |
| 训练 | `train.csv` 18353 张，`image` + `label` |
| 测试 | `test.csv` 8800 张（仅路径） |
| 类别 | **176** 种叶片 |
| 划分 | train 80% / val 20%，分层采样（`random_state=42`） |
| 图片 | `Classify Leaves/data/classify-leaves/images/*.jpg` |

> `data/` 体积大，默认不提交到 Git（见 `.gitignore`）。

## 当前进度

| 阶段 | 状态 | 产出 |
|------|------|------|
| 0 数据与脚手架 | ✅ 完成 | `scripts/dataset_common.py`、`00_eda.py`、`00_train_skeleton.py` |
| 1 LeNet | ✅ 完成 | `models/lenet.py` → `outputs/lenet.pth` |
| 2 AlexNet | ✅ 完成 | `models/alexnet.py` → `outputs/alexnet.pth` |
| 3 VGG | 🔄 进行中 | `models/vgg.py`（网络已写，训练循环待补） |
| 4 ResNet | ⏳ 未开始 | — |

## 结果对比（验证集）

| 模型 | 输入 | 要点 | 最佳 val acc | 备注 |
|------|------|------|--------------|------|
| 假模型（骨架） | 32×32 | Flatten + Linear | ~2.6%（1 epoch） | 确认链路能跑 |
| LeNet | 32×32 | 小卷积 + AvgPool | **~45.5%** | 有增强；训完再 val |
| AlexNet | 224×224 | ReLU / Dropout / 早停 | **~75.1%**（epoch 46） | `PATIENCE=10`，存最优权重 |

随机猜测约 0.6%（1/176）。加深 + 更大输入带来明显提升。

## 目录结构（Classify Leaves）

```
Classify Leaves/
├── data/classify-leaves/     # 原始数据（本地，gitignore）
├── scripts/                  # 数据与脚手架
│   ├── dataset_common.py     # LeavesDataset / LeavesTestDataset
│   ├── 00_eda.py
│   └── 00_train_skeleton.py  # 假模型跑通训练链
├── models/                   # 网络定义 + 各自的训练 / 推理入口
│   ├── lenet.py
│   ├── alexnet.py
│   └── vgg.py                # 当前阶段
└── outputs/                  # 权重、submission.csv（gitignore）
    ├── lenet.pth
    ├── alexnet.pth
    └── submission.csv
```

约定：每个架构一个 `models/*.py`，`if __name__ == "__main__"` 里复用 `dataset_common` 做训练 / 早停 / 测试集提交。换架构时尽量只改 `model` 与 `transforms`。

## 学习路线

```
0 数据与训练脚手架
    ↓
1 LeNet          最小可用 CNN
    ↓
2 AlexNet        更大输入 + ReLU/Dropout + 早停
    ↓
3 VGG            更深的 3×3 堆叠          ← 当前
    ↓
4 ResNet         残差连接（可加迁移学习）
```

### 阶段 0 — 数据与脚手架 ✅

已完成：

1. EDA：类别数 176、样本量分布  
2. `LeavesDataset`：读图、`label` ↔ id  
3. `LeavesTestDataset`：测试集无标签，返回 `(image, path)`  
4. 分层 train / val 划分  
5. 训练 / 验证骨架（假模型验证链路）

### 阶段 1 — LeNet ✅

| 要点 | 说明 |
|------|------|
| 学什么 | 卷积 → 池化 → 全连接；通道变化；展平 |
| 输入 | **32×32**，RGB |
| 训练 | SGD `lr=0.01, momentum=0.9`；翻转 + 旋转增强 |
| 产出 | `models/lenet.py`、`outputs/lenet.pth` |

### 阶段 2 — AlexNet ✅

| 要点 | 说明 |
|------|------|
| 学什么 | 224 输入、更深卷积栈、**ReLU**、**Dropout** |
| 输入 | **224×224** |
| 训练 | 同上优化器；每 epoch 验证；**早停 + 存最优权重** |
| 产出 | `models/alexnet.py`、`outputs/alexnet.pth` |

相对 LeNet：val acc 约 45% → **75%**。曾出现训过头后 loss 回升、最后一轮权重很差——说明为何要早停与按 val 存 ckpt。

### 阶段 3 — VGG 🔄（进行中）

| 要点 | 说明 |
|------|------|
| 学什么 | **多个 3×3** 代替大卷积核；「卷积堆叠 + 池化」的块化思维 |
| 当前结构 | 五段 `2×Conv3×3`（通道 64→128→256→512→512）+ AdaptiveAvgPool(7×7) + 两层 4096 FC，接近 **VGG13** 风格 |
| 输入 | 建议 **224×224**（与 AlexNet 一致，方便对比） |
| 注意 | 参数量大，必要时减小 `batch_size`；复用 AlexNet 的早停 / 存最优逻辑 |
| 对比 | 更深是否一定更好？验证集是否过拟合？ |
| 待做 | 补全 `models/vgg.py` 的训练 / 推理 `__main__`，跑出 val acc 填入上表 |
| 产出 | `models/vgg.py` + `outputs/vgg.pth` |

### 阶段 4 — ResNet ⏳

| 要点 | 说明 |
|------|------|
| 学什么 | **残差块**、捷径连接、退化问题；可精简实现 ResNet18 |
| 两条线 | (A) 从头训练；(B) ImageNet 预训练 + 改最后一层迁移（推荐收尾） |
| 产出 | `models/resnet.py` + `outputs/resnet.pth` |

## 怎么跑

```bash
cd "Classify Leaves"
conda activate california-dl

python models/lenet.py
python models/alexnet.py
python models/vgg.py          # 补完训练循环后

# DO_TRAIN = False 时：加载已有权重，只写 outputs/submission.csv
```

提交格式对齐 `data/classify-leaves/sample_submission.csv`（`image`, `label`）。

## 练习约定

- 用 **PyTorch** 手写网络结构（阶段 1–3 禁止一上来 `torchvision.models.xxx` 当黑盒；ResNet 迁移阶段再用 pretrained）。  
- 每次换架构，尽量只改 `model` 与 `transforms`，训练循环保持稳定。  
- 上表「结果对比」随实验更新：参数量、输入尺寸、最佳 val accuracy、训练耗时。
