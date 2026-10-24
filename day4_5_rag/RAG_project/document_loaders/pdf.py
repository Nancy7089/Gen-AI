from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# Get the current directory
current_dir = Path(__file__).resolve().parent

# Load the PDF file
file_path = current_dir / "OOPS.pdf"
data = PyPDFLoader(str(file_path))

# Load all pages
docs = data.load()

# Print page 15 (index 14)
print(docs[1])