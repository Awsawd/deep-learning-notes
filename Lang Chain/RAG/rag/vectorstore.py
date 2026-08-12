"""Milvus 向量库：建表 / 写入 / 连接已有 collection。"""
from rag.embeddings import get_embeddings
from langchain_milvus import Milvus
from rag import config
from rag.loader import load_and_split

def build_or_get_vectorstore(embeddings, documents=None):
    """documents 非空则入库；否则只连接已有 collection 供检索。"""
    connection_url = config.MILVUS_URL
    collection_name = config.COLLECTION_NAME
    # 若集合已存在且要重新入库：可先 drop 再写，或换个 collection 名（自己定策略）
    if documents:
        vectorstore = Milvus.from_documents(
            drop_old=True,
            documents=documents,
            embedding=embeddings,
            connection_args={"uri": connection_url},
            collection_name=collection_name,
        )
    else:
        vectorstore = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": connection_url},
            collection_name=collection_name,
        )
    return vectorstore

if __name__ == "__main__":
    emb = get_embeddings()
    chunks = load_and_split(config.KNOWLEDGE_BASE_PATH)
    vs = build_or_get_vectorstore(emb, chunks)
    print(vs.similarity_search("Docker", k = 2))
