"""
字级 BIO 对齐到 BERT token（用 offset_mapping）
运行: python scripts/03_align_bert.py
"""
import os
import sys
from pathlib import Path

# 国内访问 Hugging Face 易超时，优先走镜像（需在 import transformers / 下载前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bio import build_label_list, load_first_sample, span_to_bio

DATA_DIR = PROJECT_ROOT / "data" / "cluener"
MODEL_NAME = "bert-base-chinese"

# 本地已有缓存则离线加载，避免每次联网校验
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    print(f"已从本地缓存加载: {MODEL_NAME}")
except Exception:
    print(f"本地无完整缓存，尝试从镜像下载: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

label2id = {lab: i for i, lab in enumerate(build_label_list())}

sample = load_first_sample(DATA_DIR / "train.json")
text = sample["text"]
label = sample.get("label", {})
char_tags = span_to_bio(label, text)

enc = tokenizer(
    text,
    return_offsets_mapping=True,
    truncation=True,
    max_length=64,
)

offsets = enc["offset_mapping"]
aligned_labels = []
for start, end in offsets:
    if start == end:  # [CLS] / [SEP] 等
        aligned_labels.append(-100)
        continue
    tag = char_tags[start]
    aligned_labels.append(label2id[tag])

tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
print("text:", text)
print("--- token / offset / label_id ---")
for tok, (s, e), lab in zip(tokens, offsets, aligned_labels):
    print(tok, (s, e), lab)
