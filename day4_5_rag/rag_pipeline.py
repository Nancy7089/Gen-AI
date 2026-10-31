from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)

pdf_path = "sample_pdf.pdf"

loader = PyPDFLoader(pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "lambda_mult": 0.5
    }
)

def format_docs_with_sources(retrieved_docs):
    formatted_chunks = []

    for doc in retrieved_docs:
        page_num = doc.metadata.get("page", 0) + 1

        formatted_chunks.append(
            f"{doc.page_content}\n[Source Page: {page_num}]"
        )

    return "\n\n---\n\n".join(formatted_chunks)


prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict document QA assistant.\n"
        "Answer the user question ONLY based on the provided context below.\n"
        "If the answer is NOT present in the provided context, respond EXACTLY with:\n"
        "\"I cannot verify this from the provided documents.\"\n\n"
        "Always append the corresponding [Source Page: X] tag at the end of each cited claim.\n\n"
        "Context:\n{context}"
    ),
    ("human", "{question}")
])

llm = init_chat_model(
    "gemini-1.5-flash",
    model_provider="google_genai",
    temperature=0.0
)

rag_chain = (
    {
        "context": retriever | format_docs_with_sources,
        "question": RunnablePassthrough()
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

valid_query = "What technical skills or projects are mentioned in the document?"

print(f"=== Query 1: '{valid_query}' ===")

response_1 = rag_chain.invoke(valid_query)

print(response_1)

print("\n" + "=" * 50 + "\n")

invalid_query = "What is the capital of France and its total population?"

print(f"=== Query 2: '{invalid_query}' ===")

response_2 = rag_chain.invoke(invalid_query)

print(response_2)