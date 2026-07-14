#探索数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#读取数据
data = pd.read_csv(PROJECT_ROOT / 'data' / 'train.csv')

#查看数据
print(data.head())

#查看行数列数
print(data.shape)

print("----------------")

#查看数据类型
print(data.dtypes)

#查看缺失值
print(data.isnull().sum())

print("-------------")

#查看数据分布
print(data.describe())