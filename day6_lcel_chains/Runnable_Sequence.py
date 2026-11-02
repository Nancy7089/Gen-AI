from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv(override=True)

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words."
)

model = init_chat_model(
    "gemini-3.8-flash",
    model_provider="google_genai",
    temperature=0.3
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "Machine Learning"})

print("=== Output from RunnableSequence ===")
print(result)