#探索数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.nanops import F
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#读取数据
data = pd.read_csv(PROJECT_ROOT / 'data' / 'train.csv')

NUMERIC_FEATURES = [
    'Year built',
    'Bathrooms',
    'Full bathrooms',
    'Total interior livable area',
    'Garage spaces',
    'Elementary School Score',
    'High School Score',
    'Tax assessed value',
    'Annual tax amount',
    'Bedrooms',
    'Lot',
    'Middle School Score',
    'house_age',
    'Listed Price',
]
CAT_FEATURES = ['Type', 'City', 'Zip', 'Region']
TOP_N_CATEGORIES = 20
CURRENT_YEAR = 2026

TYPE_ALIASES = {
    'Single Family': 'SingleFamily',
}

TARGET = 'Sold Price'


def normalize_type(series: pd.Series) -> pd.Series:
    """合并 Type 列的异名写法。"""
    return series.replace(TYPE_ALIASES)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """从原始表构造建模用特征。"""
    out = df.copy()
    out['Bedrooms'] = pd.to_numeric(out['Bedrooms'], errors='coerce')
    out['house_age'] = CURRENT_YEAR - out['Year built']
    out['Type'] = normalize_type(out['Type'])
    return out


def fit_preprocess_stats(X_train: pd.DataFrame) -> dict:
    """仅用训练集估计预处理统计量。"""
    median = X_train[NUMERIC_FEATURES].median()
    filled = X_train[NUMERIC_FEATURES].fillna(median)
    stats = {
        'median': median,
        'clip_lower': filled.quantile(0.01),
        'clip_upper': filled.quantile(0.99),
        'dummy_columns': None,
    }
    for col in CAT_FEATURES:
        stats[f'top_{col}'] = X_train[col].value_counts().nlargest(TOP_N_CATEGORIES).index
    return stats


def preprocess_dataframe(X: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """对单个数据集应用清洗规则（统计量来自训练集）。"""
    out = X.copy()
    out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].fillna(stats['median'])
    out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].clip(
        lower=stats['clip_lower'],
        upper=stats['clip_upper'],
        axis=1,
    )
    for col in CAT_FEATURES:
        out[col] = out[col].where(out[col].isin(stats[f'top_{col}']), 'Other')
    return out

def to_feature_matrix(X: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """数值标准化 + 类别 One-Hot，返回模型输入矩阵。"""
    processed = preprocess_dataframe(X, stats)
    X_cat = pd.get_dummies(processed[CAT_FEATURES], prefix=CAT_FEATURES)
    if stats['dummy_columns'] is None:
        stats['dummy_columns'] = X_cat.columns
    else:
        X_cat = X_cat.reindex(columns=stats['dummy_columns'], fill_value=0)
    return pd.concat([processed[NUMERIC_FEATURES], X_cat], axis=1)

feature_cols = NUMERIC_FEATURES + CAT_FEATURES
raw_numeric = [c for c in NUMERIC_FEATURES if c != 'house_age']
raw_cols = raw_numeric + CAT_FEATURES
df = engineer_features(data[raw_cols + [TARGET]])
X = df[feature_cols]
y = df[TARGET]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
stats = fit_preprocess_stats(X_train)
X_train_mat = to_feature_matrix(X_train, stats)
X_val_mat = to_feature_matrix(X_val, stats)

model = RandomForestRegressor(
    n_estimators=100,   # 树的数量
    random_state=42,
    n_jobs=-1,          # 用所有 CPU 核
)
model.fit(X_train_mat, y_train)

y_pred = model.predict(X_val_mat)
mae = np.mean(np.abs(y_val - y_pred))
rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
print(f"Validation MAE:  ${mae:,.0f}")
print(f"Validation RMSE: ${rmse:,.0f}")