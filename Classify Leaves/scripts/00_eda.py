import pandas as pd
from pathlib import Path



PROJECT_PATH = Path(__file__).resolve().parent.parent
df = pd.read_csv(PROJECT_PATH / 'data' / 'classify-leaves' / 'train.csv')

# print(df['label'].value_counts())
# print("--------------------------------")
# #看看每类样本的图片数量
# for label in df['label'].unique():
#     print(f"Label {label}: {len(df[df['label'] == label])} images")
# print("--------------------------------")
print(df['image'].tolist())

