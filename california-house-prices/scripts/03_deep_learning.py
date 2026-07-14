import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# False=80/20 调参 + 早停；True=100% 数据训练，用于最终 submission
FULL_TRAIN = False
FULL_TRAIN_EPOCHS = 123  # 调参模式下早停 epoch，全量训练前按实际结果修改
VAL_PATIENCE = 25
MAX_TUNING_EPOCHS = 150
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


def to_model_matrix(
    X: pd.DataFrame,
    stats: dict,
    scaler: StandardScaler,
    *,
    fit_scaler: bool = False,
) -> np.ndarray:
    """数值标准化 + 类别 One-Hot，返回模型输入矩阵。"""
    processed = preprocess_dataframe(X, stats)
    if fit_scaler:
        scaler.fit(processed[NUMERIC_FEATURES])
    X_num = scaler.transform(processed[NUMERIC_FEATURES])
    X_cat = pd.get_dummies(processed[CAT_FEATURES], prefix=CAT_FEATURES).astype(np.float32)
    if stats['dummy_columns'] is None:
        stats['dummy_columns'] = X_cat.columns
    else:
        X_cat = X_cat.reindex(columns=stats['dummy_columns'], fill_value=0)
    return np.hstack([X_num, X_cat.values]).astype(np.float32)


# 读取数据
data = pd.read_csv(PROJECT_ROOT / 'data' / 'train.csv')
feature_cols = NUMERIC_FEATURES + CAT_FEATURES
raw_numeric = [c for c in NUMERIC_FEATURES if c != 'house_age']
raw_cols = raw_numeric + CAT_FEATURES
df = engineer_features(data[raw_cols + [TARGET]])

X = df[feature_cols]
y = df[TARGET]

if FULL_TRAIN:
    print(f"Mode: full train ({len(X)} samples, {FULL_TRAIN_EPOCHS} epochs)")
    X_train_data = X
    y_train_data = y
else:
    print("Mode: tuning (80/20 split + early stopping)")
    X_train_data, X_val, y_train_data, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

preprocess_stats = fit_preprocess_stats(X_train_data)
scaler = StandardScaler()
X_train_scaled = to_model_matrix(X_train_data, preprocess_stats, scaler, fit_scaler=True)
if not FULL_TRAIN:
    X_val_scaled = to_model_matrix(X_val, preprocess_stats, scaler)

input_size = X_train_scaled.shape[1]
print(f"Input size: {input_size}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

y_train_log = np.log1p(y_train_data.values)
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train_log, dtype=torch.float32).reshape(-1, 1)
if not FULL_TRAIN:
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)

batch_size = 256
train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=batch_size,
    shuffle=True,
)


class HousePriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x)


model = HousePriceModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_val_mae = float('inf')
patience_counter = 0
best_model_state = None
max_epochs = FULL_TRAIN_EPOCHS if FULL_TRAIN else MAX_TUNING_EPOCHS

for epoch in range(max_epochs):
    epoch_loss = 0.0
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / len(train_loader)
        if FULL_TRAIN:
            print(f"Epoch {epoch + 1}/{max_epochs}, Loss (log space): {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch + 1}, Loss (log space): {avg_loss:.4f}")

    if FULL_TRAIN:
        continue

    model.eval()
    with torch.no_grad():
        y_val_pred_log = model(X_val_t.to(device))
        y_pred = np.expm1(y_val_pred_log.cpu().numpy().flatten())
        y_true = y_val.values.flatten()
        val_mae = np.mean(np.abs(y_true - y_pred))

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= VAL_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

if FULL_TRAIN:
    best_model_state = model.state_dict()
    print(f"Full train finished ({max_epochs} epochs)")
else:
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        y_val_pred_log = model(X_val_t.to(device))
        y_pred = np.expm1(y_val_pred_log.cpu().numpy().flatten())
        y_true = y_val.values.flatten()

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"Validation MAE:  ${mae:,.0f}")
    print(f"Validation RMSE: ${rmse:,.0f}")

model.load_state_dict(best_model_state)
model.eval()

outputs_dir = PROJECT_ROOT / 'outputs'
outputs_dir.mkdir(exist_ok=True)
torch.save(best_model_state, outputs_dir / 'best_model.pt')
joblib.dump({
    'preprocess_stats': preprocess_stats,
    'scaler': scaler,
    'input_size': input_size,
    'raw_cols': raw_cols,
    'feature_cols': feature_cols,
}, outputs_dir / 'preprocess.joblib')
print(f"Saved model and preprocess artifacts to {outputs_dir}")


# 预测 test.csv
test_data = pd.read_csv(PROJECT_ROOT / 'data' / 'test.csv')
test_ids = test_data['Id']
X_test = engineer_features(test_data[raw_cols])
X_test_scaled = to_model_matrix(X_test, preprocess_stats, scaler)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    y_test_pred_log = model(X_test_t)
    y_test_pred = np.expm1(y_test_pred_log.cpu().numpy().flatten())

submission = pd.DataFrame({'Id': test_ids, 'Sold Price': y_test_pred})
submission.to_csv(outputs_dir / 'submission.csv', index=False)
print(f"Saved {len(submission)} predictions to outputs/submission.csv")
