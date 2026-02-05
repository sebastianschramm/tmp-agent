from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool


@tool
def search_database(query: str) -> dict:
    """Searches the database"""
    return {"value": "answer to my question is Honduras"}



llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

agent = create_agent(model=llm, tools=[search_database], middleware=[], system_prompt="", response_format=None)

with open("hitl_agent.png", "wb") as f:
    f.write(agent.get_graph().draw_mermaid_png())

response = agent.invoke({"messages": "Let's go. What is the country of most orders?"})
print(response)

