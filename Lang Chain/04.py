"""
异步调用：ainvoke / astream / 并发 gather
对比 01～03 的同步 invoke：异步适合「同时发多个请求」，总等待时间更短。
运行: python 04.py
"""
import asyncio
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

model = init_chat_model(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base=os.getenv("DEEPSEEK_BASE_URL"),
)


async def demo_ainvoke():
    """单次异步调用（不阻塞事件循环，本身仍要等 API 返回）。"""
    print("===== 1) ainvoke =====")
    msg = await model.ainvoke("用一句话介绍斐波那契数列")
    print(msg.content)


async def demo_astream():
    """异步流式：边生成边打印（体感更快）。"""
    print("\n===== 2) astream =====")
    async for chunk in model.astream("用两句话介绍质数"):
        # 不同版本字段可能是 content 或 text
        text = getattr(chunk, "content", None) or getattr(chunk, "text", "") or ""
        print(text, end="", flush=True)
    print()


async def demo_concurrent():
    """
    并发：三个短问题一起发，总耗时接近「最慢那一次」，
    而不是三次串行相加（对比 02.py 里 for 循环三次 invoke）。
    """
    print("\n===== 3) asyncio.gather 并发 =====")
    prompts = [
        "一句话解释什么是质数",
        "一句话解释什么是偶数",
        "一句话解释什么是奇数",
    ]
    t0 = time.perf_counter()
    results = await asyncio.gather(*(model.ainvoke(p) for p in prompts))
    elapsed = time.perf_counter() - t0
    for i, r in enumerate(results):
        print(f"[{i}] {r.content}")
    print(f"并发总耗时: {elapsed:.2f}s")


async def main():
    await demo_ainvoke()
    await demo_astream()
    await demo_concurrent()


if __name__ == "__main__":
    asyncio.run(main())
