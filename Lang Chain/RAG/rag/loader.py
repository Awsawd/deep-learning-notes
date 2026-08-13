"""加载知识库 md/txt 并切分。"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from rag import config
from rag.index_state import list_kb_files


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """简易字符切分（避免导入 langchain_text_splitters 触发重依赖崩溃）。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(0, end - chunk_overlap)
    return chunks


def _load_text_documents(files: list[Path]) -> list[Document]:
    docs: list[Document] = []
    for file in files:
        path = Path(file)
        if path.name.lower() == "readme.md":
            continue
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(path.resolve())},
            )
        )
    return docs


def load_and_split_files(
    files: list[Path],
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """只加载并切分给定文件列表（增量用）。"""
    chunk_size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
    chunk_overlap = (
        chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP
    )

    docs = _load_text_documents(files)
    chunks: list[Document] = []
    for doc in docs:
        parts = _split_text(doc.page_content, chunk_size, chunk_overlap)
        for part in parts:
            chunks.append(
                Document(page_content=part, metadata=dict(doc.metadata))
            )
    return chunks


def load_and_split(raw_dir: Path) -> list[Document]:
    """加载目录下全部 md/txt 并切分（全量用）。"""
    return load_and_split_files(list_kb_files(raw_dir))


if __name__ == "__main__":
    kb = config.KNOWLEDGE_BASE_PATH
    chunks = load_and_split(kb)
    print(f"加载 {kb} 下的 .md/.txt，切分为 {len(chunks)} 个 Document")
    for chunk in chunks[:10]:
        print(chunk.page_content)
        print(chunk.metadata)
        print("-" * 100)
