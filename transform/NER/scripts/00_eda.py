"""
阶段 1：CLUENER 数据探索（EDA）
运行前请先: python scripts/01_download_cluener.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "cluener"

# CLUENER 10 类实体
ENTITY_TYPES = [
    "address",
    "book",
    "company",
    "game",
    "government",
    "movie",
    "name",
    "organization",
    "position",
    "scene",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_entities(rows: list[dict]) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        label = row.get("label") or {}
        for ent_type, ent_dict in label.items():
            # ent_dict: {实体文本: [[start, end], ...]}
            for _text, spans in ent_dict.items():
                counter[ent_type] += len(spans)
    return counter


def main() -> None:
    train_path = DATA_DIR / "train.json"
    dev_path = DATA_DIR / "dev.json"
    if not train_path.exists():
        raise FileNotFoundError(
            f"未找到 {train_path}，请先运行: python scripts/01_download_cluener.py"
        )

    train = load_jsonl(train_path)
    dev = load_jsonl(dev_path) if dev_path.exists() else []

    print("=== 基本规模 ===")
    print(f"train 条数: {len(train)}")
    print(f"dev   条数: {len(dev)}")

    print("\n=== 样例（第 1 条）===")
    sample = train[0]
    print("text :", sample["text"])
    print("label:", json.dumps(sample.get("label", {}), ensure_ascii=False))

    print("\n=== 文本长度（字符数）===")
    lengths = [len(r["text"]) for r in train]
    s = pd.Series(lengths)
    print(s.describe())

    print("\n=== 训练集实体类型计数 ===")
    ent_counts = count_entities(train)
    for t in ENTITY_TYPES:
        print(f"  {t:12s}: {ent_counts.get(t, 0)}")
    print(f"  合计实体数: {sum(ent_counts.values())}")

    # 无实体句子占比
    empty = sum(1 for r in train if not r.get("label"))
    print(f"\n无标注实体的句子: {empty} / {len(train)} ({empty / len(train):.1%})")

    print("\nEDA 完成。下一步：把 span 标注转成 BIO 标签，并接 BERT tokenizer。")


if __name__ == "__main__":
    main()
