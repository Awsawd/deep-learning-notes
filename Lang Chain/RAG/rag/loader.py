"""加载 data/raw 下的 md/txt 并切分。"""
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag import config



raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"

def load_and_split(raw_dir: Path):
    """返回：List[Document]"""
    chunks = []
    files = list(raw_dir.glob("**/*.md")) + list(raw_dir.glob("**/*.txt"))
    for file in files:
        if file.name.lower() == "readme.md":
            continue
        loader = TextLoader(str(file), encoding="utf-8")
        docs = loader.load()
        chunks.extend(docs)
    
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(chunks)
    return chunks


if __name__ == "__main__":
    KNOWLEDGE_BASE_PATH_ = config.KNOWLEDGE_BASE_PATH
    chunks = load_and_split(KNOWLEDGE_BASE_PATH_)
    print(f"加载 {KNOWLEDGE_BASE_PATH_} 下的 .md/.txt，切分为 {len(chunks)} 个 Document")
    #打印前10个chunks
    for chunk in chunks[:10]:
        print(chunk.page_content)
        print(chunk.metadata)
        print("-"*100)

