import os
import requests
from dotenv import load_dotenv
from tavily import TavilyClient

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

load_dotenv(override=True)

# 1. Build RAG Pipeline

docs = PyPDFLoader("sample.pdf").load()

chunks = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
).split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(
    model
