#profile
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
from rich import print as rprint

load_dotenv(verbose=True)

model = init_chat_model(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY") ,
    api_base=os.getenv("DEEPSEEK_BASE_URL"),
    max_tokens = 100,
    max_retries = 6,
)

# rprint(model.profile)
