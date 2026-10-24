from pathlib import Path
from langchain_community.document_loaders import TextLoader

current_dir = Path(__file__).resolve().parent

file_path = current_dir / "notes.txt"

data = TextLoader(str(file_path))

docs = data.load()

print(len(docs))
print(docs[0].page_content)