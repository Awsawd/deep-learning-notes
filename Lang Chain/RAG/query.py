"""问答入口：检索 → 拼 prompt → DeepSeek。"""
from rag.qa import ask


def main() -> None:
    while True:
        question = input("问题（空行退出）: ").strip()
        if not question:
            break
        print(ask(question))


if __name__ == "__main__":
    main()
