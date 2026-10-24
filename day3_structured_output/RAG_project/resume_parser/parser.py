from pathlib import Path

from dotenv import load_dotenv
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(env_path)

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

class ResumeData(BaseModel):
    candidate_name: str = Field(description="The name written on the resume")
    skills: list[str] = Field(description="Skills listed on the resume")
    projects: list[str] = Field(description="Projects mentioned on the resume")
    ats_score_out_of_100: int = Field(
        description="A rough resume-quality score from 0 to 100",
        ge=0,
        le=100,
    )

model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
structured_model = model.with_structured_output(ResumeData)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Extract resume information accurately. "
        "Do not invent details. If the name is missing, use 'Not found'. "
        "If skills or projects are missing, return empty lists. "
        "Estimate the score from 0 to 100 based on clarity, skills, "
        "projects, and evidence of impact. This is only a rough estimate, "
        "not a score from a real ATS system."
    ),
    (
        "human",
        "Read this resume and extract the requested information:\n\n{resume_text}"
    ),
])

project_root = Path(__file__).resolve().parents[3]
load_dotenv(project_root / ".env")

pdf_path = project_root / "day4_5_rag" / "sample_pdf.pdf"

loader = PyPDFLoader(str(pdf_path))
pages = loader.load()

resume_text = "\n\n".join(page.page_content for page in pages)

print(f"Loaded {len(pages)} pages.")
print("Text preview:")
print(resume_text[:1000])

chain = prompt | structured_model
result = chain.invoke({"resume_text": resume_text})

print("\nExtracted resume details:")
print(result.model_dump_json(indent=2))