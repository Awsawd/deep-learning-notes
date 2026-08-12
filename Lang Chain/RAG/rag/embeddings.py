"""本地嵌入模型（BGE，1024 维）。首次运行会下载权重。"""
import os

# 必须在 import huggingface / sentence_transformers 之前设置
os.environ.setdefault("HF_HOME", r"D:\huggingface")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from langchain_huggingface import HuggingFaceEmbeddings
from rag import config


def get_embeddings():
    """返回嵌入器。"""
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings

if __name__ == "__main__":
    emb = get_embeddings()
    vec = emb.embed_query("测试")
    print(len(vec))
    assert len(vec) == config.EMBED_DIM
