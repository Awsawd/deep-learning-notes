#temperature
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv


load_dotenv(verbose=True)

model = init_chat_model(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,                         #
)

for i in range(3):
    print(model.invoke("写一首诗").content)