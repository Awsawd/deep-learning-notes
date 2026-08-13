"""检索 TopK + 拼 prompt + DeepSeek 生成。"""
from rag import config
from rag.embeddings import get_embeddings
from rag.vectorstore import build_or_get_vectorstore
from langchain.chat_models import init_chat_model

SYSTEM = (
    "你是助手。请仅根据【上下文】回答；若上下文不足，请明确说不知道。"
    "回答末尾列出用到的来源文件名。"
)


def build_prompt(question: str, docs: list) -> str:
    """把 TopK 文档拼成带上下文的用户消息。"""
    shangxiawen = "\n\n".join([doc.page_content for doc in docs])
    sources = [doc.metadata.get("source") for doc in docs]
    return f"""
    上下文：{shangxiawen}
    来源：{sources}
    问题：{question}
    """
    
def ask(question: str, *, k: int = 4) -> str:
    """similarity_search → build_prompt → init_chat_model(DeepSeek) → invoke。"""
    emb = get_embeddings()
    vs = build_or_get_vectorstore(emb)
    docs = vs.similarity_search(question, k = k)
    user = build_prompt(question, docs)
    model = init_chat_model(
        model="deepseek-v4-flash",
        api_key=config.DEEPSEEK_API_KEY,
        api_base=config.DEEPSEEK_BASE_URL,
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    return model.invoke(messages).content

if __name__ == "__main__":
    print(ask("Docker 是什么？"))