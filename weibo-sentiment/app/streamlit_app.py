"""
Streamlit Demo：输入中文文本 → 正/中/负 情感 + 概率
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "outputs" / "bert-sentiment" / "best"
ID2LABEL = {0: "负面", 1: "中性", 2: "正面"}


@st.cache_resource
def load_model(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tok, model, device


def predict(text: str, tok, model, device):
    enc = tok(text, truncation=True, max_length=128, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0]
    pred_id = int(probs.argmax().item())
    return ID2LABEL.get(pred_id, str(pred_id)), {ID2LABEL[i]: float(probs[i]) for i in range(len(probs))}


def main():
    st.set_page_config(page_title="微博舆情情感分析 Demo", layout="centered")
    st.title("舆情情感分析 Demo")
    st.caption("BERT 三分类（负面 / 中性 / 正面）。训练语料为公开数据集；微博数据用于采集与趋势展示。")

    model_dir = st.text_input("模型目录", value=str(DEFAULT_MODEL))
    if not Path(model_dir).exists():
        st.error(f"模型不存在: {model_dir}。请先运行 `python scripts/03_train_bert.py`。")
        st.stop()

    tok, model, device = load_model(model_dir)
    text = st.text_area("输入文本", value="这个产品体验真的很不错，会回购。", height=120)
    if st.button("分析", type="primary") and text.strip():
        label, probs = predict(text.strip(), tok, model, device)
        st.subheader(f"预测：{label}")
        st.bar_chart(probs)

    st.divider()
    st.markdown(
        """
**说明**
- 仅供学习演示，结果不构成任何舆情结论。
- Cookie / 原始评论请勿公开上传。
- 项目流水线见仓库 `weibo-sentiment/README.md`。
"""
    )


if __name__ == "__main__":
    main()
