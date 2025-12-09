"""
Simple Invoking the LLM.
"""

# from langchain_openai import OpenAI
from langchain_ollama.chat_models import ChatOllama
from dotenv import load_dotenv

load_dotenv(".env")
import os

# print(os.getenv("OPENAI_API_KEY"))
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# result = client.invoke("Hello")
# print(result)

llm = ChatOllama(model="qwen3:1.7b")
print(llm.invoke("Hello!!"))