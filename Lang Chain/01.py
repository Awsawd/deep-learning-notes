from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model

load_dotenv(override=True)
#
# llm_deepseek = ChatDeepSeek(
#     model="deepseek-v4-flash",
#     api_key=os.getenv("DEEPSEEK_API_KEY") ,
#     api_base=os.getenv("DEEPSEEK_BASE_URL"),
# )


# response = llm_deepseek.invoke("Hello, how are you?")
# print(response.content)

model = init_chat_model(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY") ,
    api_base=os.getenv("DEEPSEEK_BASE_URL"),
)

# response = model.invoke("Hello, how are you?")
# print(response.content)

