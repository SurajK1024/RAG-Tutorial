# Step 1: Load + Chunk PDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

loader = PyPDFLoader("data/tesla-model-s.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

# Step 2: Create Embeddings + Store

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

embedding_model = OllamaEmbeddings(model="qwen3-embedding:4b")

vectorstore = FAISS.from_documents(chunks, embedding_model)

vectorstore.save_local("faiss_index")

# Step 3: Query + Retrieve

query = "What is the main topic?"

docs = vectorstore.similarity_search(query, k=3)

context = "\n".join([doc.page_content for doc in docs])

# Step 4: Generate Answer (LLM)

from langchain_ollama.chat_models import ChatOllama

llm = ChatOllama(model="qwen3:1.7b")

prompt = f"""
Answer based only on the context below:

{context}

Question: {query}
"""

response = llm.invoke(prompt)

print(response)