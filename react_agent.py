from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


@tool
def search_db(query: str) -> str:
    """Search the database"""
    return "foobar: 100"


tools = [search_db]

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


agent = create_agent(model, tools)
response = agent.invoke({"messages": [HumanMessage("How many orders are in the db?")]})


print(response)
print()
print(response["messages"][-1])
breakpoint()
