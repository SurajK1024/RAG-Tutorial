# %%
# %pip install langchain langchain-community langchain-openai faiss-cpu pymupdf beautifulsoup4 python-dotenv

# %%
# If not installed

from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai.chat_models import ChatOpenAI
# from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
import os

# %%
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# %%
def chatAndEmbedModel():
    embed_model = OllamaEmbeddings(model="qwen3-embedding:4b")
    chat_model = ChatOllama(model="qwen3:1.7b")
    return chat_model, embed_model

chat_model, embed_model = chatAndEmbedModel()

# %%
def docsloader(source, pdf=True):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    if pdf:
        loader = PyMuPDFLoader(source)
    else:
        loader = WebBaseLoader(source)

    docs = loader.load_and_split(text_splitter)
    return docs

# %%
pdfPath = "data/tesla-model-s.pdf"

pdfdocs = docsloader(pdfPath)

print(f"Loaded {len(pdfdocs)} PDF chunks")
print(pdfdocs[0].page_content)

# %%
urlPath = "https://www.carwale.com/porsche-cars/718/cayman/"

urldocs = docsloader(urlPath, pdf=False)

print(f"Loaded {len(urldocs)} URL chunks")
print(urldocs[0].page_content)

# %%
all_docs = pdfdocs + urldocs

vectorDB = FAISS.from_documents(all_docs, embed_model)

print("Vector DB created successfully")

# %%
query = "What is the range of Tesla Model S?"

results = vectorDB.similarity_search(query, k=3)

for i, r in enumerate(results):
    print(f"\nResult {i+1}:\n")
    print(r.page_content)

# %%
retriever = vectorDB.as_retriever()

docs = retriever.invoke("Tell me about Porsche Cayman")

for i, doc in enumerate(docs):
    print(f"\nDoc {i+1}:\n")
    print(doc.page_content)

# %%
vectorDB.save_local("faiss_index")

# %%