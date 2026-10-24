from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

project_root = Path(__file__).resolve().parents[3]
load_dotenv(project_root / ".env")

pdf_path = Path(__file__).resolve().parent / "Nancy_Resume.pdf"

loader = PyPDFLoader(str(pdf_path))
pages = loader.load()

resume_text = "\n\n".join(page.page_content for page in pages)

print(f"Loaded {len(pages)} pages.")
print("Text preview:")
print(resume_text[:1000])