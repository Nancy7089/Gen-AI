from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

pdf_path = Path(__file__).resolve().parent / "sample_pdf.pdf"

loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

splitter_1 = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0
)
chunks_1 = splitter_1.split_documents(docs)
vector_store_1 = FAISS.from_documents(chunks_1, embeddings)

splitter_2 = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
chunks_2 = splitter_2.split_documents(docs)
vector_store_2 = FAISS.from_documents(chunks_2, embeddings)

query = "What is covered in the bootcamp plan"

print("=== CONFIG 1 RESULTS (200 / 0) ===")
results_1 = vector_store_1.similarity_search(query, k=2)
for i, doc in enumerate(results_1, start=1):
    print(f"\n[Chunk {i}]:\n{doc.page_content}")

print("\n" + "=" * 40 + "\n")

print("=== CONFIG 2 RESULTS (800 / 150) ===")
results_2 = vector_store_2.similarity_search(query, k=2)
for i, doc in enumerate(results_2, start=1):
    print(f"\n[Chunk {i}]:\n{doc.page_content}")