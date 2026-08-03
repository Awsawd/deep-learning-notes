"""BIO 标签与 CLUENER span 转换（可被其它脚本正常 import）。"""
from __future__ import annotations

import json
from pathlib import Path

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


def build_label_list() -> list[str]:
    labels = ["O"]
    for t in ENTITY_TYPES:
        labels.append(f"B-{t}")
        labels.append(f"I-{t}")
    return labels


def span_to_bio(label: dict, text: str) -> list[str]:
    """CLUENER 的 [start, end] 按闭区间处理。"""
    tags = ["O"] * len(text)
    if not label:
        return tags

    for ent_type, ent_dict in label.items():
        for _ent_text, spans in ent_dict.items():
            for start, end in spans:
                if start < 0 or end >= len(text) or start > end:
                    continue
                tags[start] = f"B-{ent_type}"
                for i in range(start + 1, end + 1):
                    tags[i] = f"I-{ent_type}"
    return tags


def load_first_sample(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.loads(f.readline())
