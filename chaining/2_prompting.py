"""
Invoking the LLM using Some prompts
"""

from langchain_openai import OpenAI
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

prompt = PromptTemplate.from_template(
"""
Acts as an Medicial professional and give advice for the user queries.
Provide set of instructions the user need to follow to prevent the cause or the disease.
Generate answer in such format:
Disease cause: 
Medicine name:
In-take time:
Prevention method:

User Query:{user_query}
"""
)

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
user_query = input()
#result = client.invoke(prompt.format(user_query=user_query))
#print(result)

llm = ChatOllama(model="qwen3:1.7b")
print(
    llm.invoke(
        prompt.format(
            user_query=user_query
            )
        )
    )