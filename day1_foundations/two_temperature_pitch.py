from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

prompt="Give me a 1-line pitch for a food delivery app."

for temperature in(0.0,1.2):
    llm=ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=temperature,
    )
    response=llm.invoke(prompt)
    print(f"\n--- temperature={temperature} ---")
    content = response.content

if isinstance(content, str):
    response_text = content
else:
    response_text = "\n".join(
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )

    print("Response:", response_text)

    usage = response.usage_metadata or {}
    print("Input tokens:", usage.get("input_tokens", "not reported"))
    print("Output tokens:", usage.get("output_tokens", "not reported"))
