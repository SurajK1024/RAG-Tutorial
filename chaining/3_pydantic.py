from langchain_openai import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

class ResponseModel(BaseModel):
    Country: str = Field(description="The country name provided by the user.")
    Place: list[str] = Field(description="List of places to visit or experience.")
    Food: list[str] = Field(description="Foods to try in each place.")
    Cloth: list[str] = Field(description="List of clothes to take.")
    Must: list[str] = Field(description="MUST DO activities in this country.")
    Instruction: str | None = Field(default=None, description="Important safety & travel instructions.")

prompt = PromptTemplate.from_template("""
Act as a Travel Instructor.
User Query: {user_query}
""")

# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0,
#     api_key=os.getenv("OPENAI_API_KEY")
# )

llm = ChatOllama(model="qwen3:1.7b")

structured_llm = llm.with_structured_output(ResponseModel)

result = structured_llm.invoke(prompt.format(user_query="Tell me about New Zealand!!!"))
print(result)