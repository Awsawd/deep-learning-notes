from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from openai.types.beta import assistant

load_dotenv(override=True)

model = init_chat_model(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY") ,
    api_base=os.getenv("DEEPSEEK_BASE_URL"),
)

conversation = [
    {"role":"system","content":"你是一个专业的数学老师"},
    {"role":"user","content":"帮我解释一下什么是斐波拉契数列"},
]

response1 = model.invoke(conversation)
print(response1)

#添加记忆
conversation.append({"role":"assistant","content":response1.content})
conversation.append({"role":"user","content":"我刚刚问了什么问题"})

response2 = model.invoke(conversation)
print(response2)

"""
for chunk in model.stream("随便回一句话"):
    print(chunk.text,end="",flush=True))
"""

