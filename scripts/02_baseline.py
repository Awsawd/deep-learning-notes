#探索数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.nanops import F
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#读取数据
data = pd.read_csv(PROJECT_ROOT / 'data' / 'train.csv')

# 定义特征列和目标变量
FEATURES = [
    'Year built',
    'Bathrooms',
    'Full bathrooms',
    'Total interior livable area',
    'Garage spaces',
    'Elementary School Score',
    'High School Score',
    'Tax assessed value',
    'Annual tax amount'
]

TARGET = 'Sold Price'

# 从 data 里取出特征和目标变量
df = data[FEATURES + [TARGET]]

# 打印 shape
# print(df.shape)

##缺失值处理

#查看缺失值
# print(df[FEATURES].isnull().sum())
# print("-------------")

#中位值填充
df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

# print(df[FEATURES].isnull().sum())

lower_limit = df['Total interior livable area'].quantile(0.01)
upper_limit = df['Total interior livable area'].quantile(0.99)
df['Total interior livable area'] = df['Total interior livable area'].clip(lower=lower_limit, upper=upper_limit)

#划分训练集和验证集
X_train, X_val = train_test_split(df, test_size=0.2, random_state=42)
y_train = X_train[TARGET]
y_val = X_val[TARGET]

#打印训练集和验证集的shape
print(X_train.shape)
print(X_val.shape)
print("-------------")
print(y_train.shape)
print(y_val.shape)