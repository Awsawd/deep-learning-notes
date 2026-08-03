"""字级 BIO 自测：python scripts/02_span_to_bio.py"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bio import build_label_list, load_first_sample, span_to_bio

DATA_DIR = PROJECT_ROOT / "data" / "cluener"

if __name__ == "__main__":
    sample = load_first_sample(DATA_DIR / "train.json")
    text = sample["text"]
    label = sample.get("label", {})
    tags = span_to_bio(label, text)
    print("标签表大小:", len(build_label_list()))
    print("text:", text)
    print("len(text), len(tags):", len(text), len(tags))
    print("--- 逐字对照（前 25 个）---")
    for ch, tag in list(zip(text, tags))[:25]:
        print(repr(ch), tag)
