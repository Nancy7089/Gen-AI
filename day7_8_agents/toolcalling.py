from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool 
from langchain_core.messages import HumanMessage
from rich import print 

# 1. Creating a tool 
@tool
def get_text_length(text: str) -> int:
    """Returns the number of character in a given text"""
    return len(text)

tools = {
    "get_text_length" : get_text_length
}

llm = ChatMistralAI(model="mistral-small-2506")

# 2. Tool Binding 
llm_with_tool = llm.bind_tools([get_text_length])

messages = []
prompt = input("You: ")
messages.append(HumanMessage(content=prompt))

# 3. The Manual Agent Execution Loop 

while True:
    result = llm_with_tool.invoke(messages)
    
    messages.append(result)
    
    if not result.tool_calls:
        print(result.content)
        break
        
    for tool_call in result.tool_calls:
        tool_name = tool_call["name"]
        print(f"[System] Executing tool: {tool_name}...")
        
        tool_message = tools[tool_name].invoke(tool_call)
        
        messages.append(tool_message)