from dotenv import load_dotenv
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv(override=True)


@tool
def calculate_emi(principal: float, rate: float, months: int) -> str:
    """
    Calculate the monthly EMI for a loan.

    Args:
        principal: Total loan amount.
        rate: Annual interest rate in percentage.
        months: Loan tenure in months.
    """
    monthly_rate = (rate / 12) / 100

    if monthly_rate == 0:
        emi = principal / months
    else:
        emi = (
            principal * monthly_rate * ((1 + monthly_rate) ** months)
            / (((1 + monthly_rate) ** months) - 1)
        )

    return (
        f"Estimated monthly EMI is {emi:.2f} "
        f"for principal {principal} over {months} months "
        f"at {rate}% annual interest."
    )


@tool
def check_loan_eligibility(income: float, credit_score: int) -> str:
    """
    Check loan eligibility based on monthly income and credit score.

    Args:
        income: Applicant's monthly income.
        credit_score: Applicant's credit score (300 to 900).
    """
    if credit_score >= 700 and income >= 25000:
        max_loan = income * 20
        return (
            f"Eligible! With income {income} and credit score "
            f"{credit_score}, estimated maximum loan is {max_loan:.2f}."
        )

    elif credit_score >= 650 and income >= 15000:
        max_loan = income * 10
        return (
            f"Conditionally eligible. Estimated maximum loan is "
            f"{max_loan:.2f}. Higher interest rates may apply."
        )

    else:
        return (
            f"Not eligible. Income ({income}) or credit score "
            f"({credit_score}) does not meet the minimum criteria."
        )


tools_list = [calculate_emi, check_loan_eligibility]

tools_map = {t.name: t for t in tools_list}


llm = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0.0
)

llm_with_tools = llm.bind_tools(tools_list)


print("=== Finance AI Agent Initialized ===")
print("Type 'exit' or 'quit' to stop.")

messages = []

while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    if not user_input:
        continue

    messages.append(HumanMessage(content=user_input))

    step = 1

    while True:
        print(f"\n[Step {step}] Invoking LLM...")

        result = llm_with_tools.invoke(messages)
        messages.append(result)

        if result.tool_calls:
            print(
                f"[Step {step}] LLM requested "
                f"{len(result.tool_calls)} tool call(s)."
            )

            for tool_call in result.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                print(
                    f" -> Calling Tool: '{tool_name}' "
                    f"with args: {tool_args}"
                )

                selected_tool = tools_map[tool_name]
                tool_output = selected_tool.invoke(tool_args)

                print(f" <- Tool Output: {tool_output}")

                messages.append(
                    ToolMessage(
                        content=str(tool_output),
                        tool_call_id=tool_call["id"]
                    )
                )

            step += 1
            continue

        else:
            print(f"[Step {step}] Final Answer Ready.")
            print(f"\nAI: {result.content}")
            break
