# California House Prices

个人学习项目：加州房价预测（EDA → 传统 ML 基线 → PyTorch MLP）。

## 环境

- Python 3.11
- Conda 环境：`california-dl`（PyTorch GPU + pandas + scikit-learn）
- 解释器路径示例：`H:\conda_envs\california-dl\python.exe`

## 项目结构

```
data/                  原始 CSV 数据
scripts/
  01_eda.py            数据探索
  02_baseline.py       数据准备 / sklearn 基线（进行中）
  03_deep_learning.py  PyTorch MLP（log1p + Type One-Hot）
outputs/               图表输出（本地生成，不入库）
```

## 运行

```bash
conda activate california-dl
python scripts/03_deep_learning.py
```

## 数据

来自 Kaggle 风格加州房价数据集：`train.csv`、`test.csv`、`sample_submission.csv`。
