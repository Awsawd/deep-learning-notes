"""
清洗 data/raw/*.jsonl → data/cleaned/comments.parquet

去 URL / @用户 / 多余空白，按 content 去重，过滤过短过长，并打印 EDA。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned"
OUT_PATH = CLEAN_DIR / "comments.parquet"

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
AT_RE = re.compile(r"@[\w\u4e00-\u9fff\-]+")
TOPIC_RE = re.compile(r"#[^#\s]+#")
SPACE_RE = re.compile(r"\s+")
EMOJI_TAG_RE = re.compile(r"\[[^\[\]]{1,12}\]")  # 微博表情 [微笑]


def clean_text(s: str) -> str:
    s = str(s)
    s = URL_RE.sub(" ", s)
    s = AT_RE.sub(" ", s)
    s = EMOJI_TAG_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s


def load_raw_jsonl(raw_dir: Path) -> pd.DataFrame:
    rows = []
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"未找到原始文件: {raw_dir}/*.jsonl ，请先运行 01_crawl_weibo.py")
    for fp in files:
        if fp.name.startswith("_"):
            continue
        with fp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-len", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=280)
    args = parser.parse_args()

    df = load_raw_jsonl(RAW_DIR)
    print("原始条数:", len(df), "列:", list(df.columns))
    if "content" not in df.columns:
        raise ValueError("原始数据缺少 content 字段")

    df["content_raw"] = df["content"].astype(str)
    df["content"] = df["content_raw"].map(clean_text)
    df["text_len"] = df["content"].str.len()
    df = df[(df["text_len"] >= args.min_len) & (df["text_len"] <= args.max_len)].copy()
    df["content_hash"] = df["content"].map(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())
    before = len(df)
    df = df.drop_duplicates(subset=["content_hash"]).reset_index(drop=True)
    print(f"去重: {before} -> {len(df)}")

    # 解析时间（失败则保留空）
    if "created_at" in df.columns:
        df["created_at_parsed"] = pd.to_datetime(df["created_at"], errors="coerce")
        df["date"] = df["created_at_parsed"].dt.date.astype("string")
    else:
        df["date"] = pd.NA

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    keep = [c for c in ["id", "user", "content", "content_raw", "created_at", "date", "status_url", "source", "text_len"] if c in df.columns]
    out = df[keep]
    out.to_parquet(OUT_PATH, index=False)
    print("已保存:", OUT_PATH)
    print("清洗后条数:", len(out))
    print("长度 describe:\n", out["text_len"].describe())
    if "source" in out.columns:
        print("source 分布:\n", out["source"].value_counts())


if __name__ == "__main__":
    main()
