"""
对 data/cleaned/comments.parquet 批量情感推理 → data/predicted/comments_pred.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "cleaned" / "comments.parquet"
PRED_DIR = PROJECT_ROOT / "data" / "predicted"
MODEL_DIR = PROJECT_ROOT / "outputs" / "bert-sentiment" / "best"
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
ZH_LABEL = {"negative": "负面", "neutral": "中性", "positive": "正面"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR))
    args = parser.parse_args()

    if not CLEAN_PATH.exists():
        raise SystemExit(f"缺少清洗数据: {CLEAN_PATH}，请先运行 02_clean.py")
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"缺少模型: {model_dir}，请先运行 03_train_bert.py")

    df = pd.read_parquet(CLEAN_PATH)
    texts = df["content"].astype(str).tolist()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    labels, probs, confs = [], [], []
    bs = args.batch_size
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            enc = tokenizer(batch, truncation=True, max_length=128, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1)
            pred = p.argmax(dim=-1)
            for j in range(len(batch)):
                lid = int(pred[j].item())
                lab = ID2LABEL.get(lid, str(lid))
                labels.append(lab)
                confs.append(float(p[j, lid].item()))
                probs.append({ID2LABEL[k]: float(p[j, k].item()) for k in range(p.shape[1])})
            if (i // bs) % 20 == 0:
                print(f"推理进度 {min(i + bs, len(texts))}/{len(texts)}")

    df["pred_label"] = labels
    df["pred_label_zh"] = df["pred_label"].map(lambda x: ZH_LABEL.get(x, x))
    df["pred_confidence"] = confs
    df["pred_probs"] = probs

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = PRED_DIR / "comments_pred.parquet"
    df.to_parquet(out, index=False)
    print("已保存:", out)
    print(df["pred_label_zh"].value_counts())


if __name__ == "__main__":
    main()
