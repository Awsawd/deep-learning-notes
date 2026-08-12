"""入库入口：加载 → 切分 → 嵌入 → 写入 Milvus。"""
from rag.config import KNOWLEDGE_BASE_PATH
from rag.embeddings import get_embeddings
from rag.loader import load_and_split
from rag.vectorstore import build_or_get_vectorstore


def main() -> None:
    docs = load_and_split(KNOWLEDGE_BASE_PATH)
    embeddings = get_embeddings()
    build_or_get_vectorstore(embeddings, documents=docs)
    print(f"ingested chunks: {len(docs)}")


if __name__ == "__main__":
    main()
