"""Milvus 向量库：建表 / 写入 / 连接 / 按 source 删除。"""
from __future__ import annotations

import json

from langchain_core.documents import Document
from langchain_milvus import Milvus

from rag import config


def build_or_get_vectorstore(embeddings, documents: list[Document] | None = None):
    """documents 非空则全量重建入库；否则只连接已有 collection。"""
    connection_args = {"uri": config.MILVUS_URL}
    collection_name = config.COLLECTION_NAME

    if documents:
        return Milvus.from_documents(
            documents=documents,
            embedding=embeddings,
            connection_args=connection_args,
            collection_name=collection_name,
            drop_old=True,
        )

    return Milvus(
        embedding_function=embeddings,
        connection_args=connection_args,
        collection_name=collection_name,
    )


def delete_by_source(vs: Milvus, source: str) -> None:
    """按 metadata.source 删除该文件对应的全部向量。"""
    # json.dumps 正确转义 Windows 路径中的反斜杠与引号
    expr = f"source == {json.dumps(source, ensure_ascii=False)}"
    try:
        vs.delete(expr=expr)
    except Exception as exc:  # noqa: BLE001 — 新文件或字段尚未建立时允许跳过
        print(f"delete_by_source skip ({source}): {exc}")


if __name__ == "__main__":
    from rag.embeddings import get_embeddings
    from rag.loader import load_and_split

    emb = get_embeddings()
    chunks = load_and_split(config.KNOWLEDGE_BASE_PATH)
    vs = build_or_get_vectorstore(emb, chunks)
    print(vs.similarity_search("Docker", k=2))
