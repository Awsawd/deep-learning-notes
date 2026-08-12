"""环境自检：Milvus 连通、配置是否可读（嵌入维数可放到 embeddings 写完后再测）。"""
import sys
from dotenv import load_dotenv
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pymilvus import MilvusClient
from rag import config

def check_config() -> None:
    print(f"MILVUS_URI: {config.MILVUS_URI}")
    print(f"EMBED_DIM: {config.EMBED_DIM}")
    print(f"KNOWLEDGE_BASE_PATH: {config.KNOWLEDGE_BASE_PATH}")
    if not config.DEEPSEEK_API_KEY:
        print(f"DEEPSEEK_API_KEY 不存在")
    else:
        print(f"DEEPSEEK_API_KEY 存在")

def check_milvus():
    client = MilvusClient(uri=config.MILVUS_URI)
    try:
        names = client.list_collections()
        print(f"Milvus 连接成功，集合列表: {names}")
    finally:
        client.close()

def main() -> None:
    # 1) 从 rag.config 打印 MILVUS_URI、EMBED_DIM、KNOWLEDGE_BASE_PATH、DEEPSEEK 是否有值
    check_config()
    # 2) 用 pymilvus.MilvusClient(uri=...) 连接，list_collections()
    check_milvus()
    # 3) （可选）调用 get_embeddings()，assert 向量长度 == EMBED_DIM


if __name__ == "__main__":
    main()
