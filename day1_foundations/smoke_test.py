from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.0
)

response = llm.invoke("Reply with exactly: Codespace is working.")

content = response.content

if isinstance(content, str):
    print(content)
elif isinstance(content, list):
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            print(block.get("text", ""))
        elif isinstance(block, str):
            print(block)
else:
    print(content)