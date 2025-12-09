### Create RAG and Runnable to automate it

from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai.chat_models import ChatOpenAI
# ̉from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings

from dotenv import load_dotenv
import os

load_dotenv()

def chatAndEmbedModel():
    embed_model = OllamaEmbeddings(model="qwen3-embedding:4b")
    chat_model = ChatOllama(model="qwen3:1.7b")

    return chat_model, embed_model

def docsloader(url, pdf=True):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", ""])

    if pdf:
        loader = PyMuPDFLoader(url)
    else:
        loader = WebBaseLoader(url)

    print("~"*100)
    docs = loader.load_and_split(text_splitter)
    return docs

if __name__ == "__main__":
    # chat_model, embed_model = chatAndEmbedModel()
    pdfPath = "data/tesla-model-s.pdf"
    pdfdocs = docsloader(pdfPath)

    print("PDF Docs loader")
    print(pdfdocs)

    urlPath = "https://www.carwale.com/porsche-cars/718/cayman/"
    urldocs = docsloader(urlPath, pdf=False)

    print("URL Docs loader")
    print(urldocs)