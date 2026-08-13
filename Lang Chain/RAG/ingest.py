"""入库入口：默认增量；--full 全量重建。"""
from __future__ import annotations

import argparse

from rag import config
from rag.embeddings import get_embeddings
from rag.index_state import (
    STATE_PATH,
    diff_files,
    file_sha256,
    list_kb_files,
    load_state,
    save_state,
)
from rag.loader import load_and_split, load_and_split_files
from rag.vectorstore import build_or_get_vectorstore, delete_by_source


def _fresh_state_files() -> dict:
    return {
        str(path): {"sha256": file_sha256(path)}
        for path in list_kb_files(config.KNOWLEDGE_BASE_PATH)
    }


def ingest_full() -> None:
    """删除旧 collection，全量写入，并重写 index_state。"""
    print(f"知识库: {config.KNOWLEDGE_BASE_PATH}")
    docs = load_and_split(config.KNOWLEDGE_BASE_PATH)
    embeddings = get_embeddings()
    build_or_get_vectorstore(embeddings, documents=docs)

    state = {
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "files": _fresh_state_files(),
    }
    save_state(state)
    print(f"全量完成: chunks={len(docs)}, files={len(state['files'])}")
    print(f"状态文件: {STATE_PATH}")


def ingest_incremental() -> None:
    """只处理新增/变更/删除的文件。"""
    if not STATE_PATH.exists():
        print("未找到 index_state.json，自动改为全量入库…")
        ingest_full()
        return

    state = load_state()
    pending, deleted = diff_files(config.KNOWLEDGE_BASE_PATH, state)
    print(f"知识库: {config.KNOWLEDGE_BASE_PATH}")
    print(f"pending={len(pending)}, deleted={len(deleted)}")

    if not pending and not deleted:
        print("无变更，跳过。")
        return

    embeddings = get_embeddings()
    vs = build_or_get_vectorstore(embeddings)  # 只连接，不 drop

    for source in deleted:
        delete_by_source(vs, source)
        state["files"].pop(source, None)
        print(f"已删除: {source}")

    for path in pending:
        key = str(path.resolve())
        delete_by_source(vs, key)
        chunks = load_and_split_files([path])
        if chunks:
            vs.add_documents(chunks)
        state["files"][key] = {"sha256": file_sha256(path)}
        print(f"已更新: {key} (chunks={len(chunks)})")

    state["embedding_model"] = config.EMBEDDING_MODEL_NAME
    state["chunk_size"] = config.CHUNK_SIZE
    state["chunk_overlap"] = config.CHUNK_OVERLAP
    save_state(state)
    print(f"增量完成。状态文件: {STATE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 知识库入库")
    parser.add_argument(
        "--full",
        action="store_true",
        help="全量重建（drop collection + 重写 state）",
    )
    args = parser.parse_args()
    if args.full:
        ingest_full()
    else:
        ingest_incremental()


if __name__ == "__main__":
    main()
