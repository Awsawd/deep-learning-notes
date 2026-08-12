"""从 .env 读取配置。可按需自己加校验（空字符串、路径是否存在等）。"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env")  # 复用上级 DeepSeek；注意空字符串会挡住默认值

MILVUS_URL = os.getenv("MILVUS_URL", "http://localhost:19530")
DB_NAME = os.getenv("DB_NAME", "RAG_tutorial")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "doc")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "BAAI/bge-large-zh-v1.5"
)
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
KNOWLEDGE_BASE_PATH = Path(
    os.getenv("KNOWLEDGE_BASE_PATH", str(ROOT / "data" / "raw"))
)
if not KNOWLEDGE_BASE_PATH.is_absolute():
    KNOWLEDGE_BASE_PATH = (ROOT / KNOWLEDGE_BASE_PATH).resolve()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
