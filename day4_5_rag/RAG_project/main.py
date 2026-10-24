from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from the project root
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Load the text file
file_path = Path(__file__).resolve().parent.parent / "document_loaders" / "notes.txt"

data = TextLoader(str(file_path))
docs = data.load()

# Create the prompt template
template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI that summarizes the given text."),
    ("human", "{data}")
])

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

# Format the prompt with the document content
prompt = template.format_messages(data=docs[0].page_content)

# Get the model response
result = model.invoke(prompt)

# Print the summary
print(result.content)