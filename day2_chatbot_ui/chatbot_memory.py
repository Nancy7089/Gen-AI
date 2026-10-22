from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv(override=True)

llm = init_chat_model("gemini-3.8-flash", model_provider="google_genai", temperature=0.7)

system_message = SystemMessage(content="You are a helpful and concise AI assistant.")

messages = [system_message]

print("=== Stateful Chatbot Initialized (Type 'exit' or 'quit' to stop) ===")

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    
    if not user_input:
        continue

    messages.append(HumanMessage(content=user_input))

    trimmed_messages = [system_message] + messages[1:][-6:]

    response = llm.invoke(trimmed_messages)

    messages.append(AIMessage(content=response.content))
    print(f"AI: {response.content}")

    print(f"   [Debug: Active context sent to LLM = {len(trimmed_messages)} messages]")