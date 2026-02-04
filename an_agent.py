from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


@tool
def add_numbers(number_a: float, number_b: float) -> float:
    """
    Adds two float numbers and returns the result as a float.
    """
    return number_a + number_b

my_tools = [add_numbers]


llm_name = "gemini-2.5-flash"

llm = ChatGoogleGenerativeAI(model=llm_name, temperature=0.0)
llm = llm.bind_tools(my_tools)



prompt = "What is the sum 2 and 5?"

response = llm.invoke(prompt)

tc = response.tool_calls[0]
print(add_numbers.invoke(tc))
print(response)

