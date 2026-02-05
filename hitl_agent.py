from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import AgentState


checkpointer = MemorySaver()


class CustomState(AgentState):
    user_preferences: dict
    number_of_tool_calls: int


@tool
def search_database(query: str) -> dict:
    """Searches the database"""
    return {"value": "answer to my question is Honduras"}


#llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.0)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

agent = create_agent(
    model=llm,
    tools=[search_database],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "search_database": {"allowed_decisions": ["approve", "reject"]}
            }
        )
    ],
    system_prompt="",
    response_format=None,
    checkpointer=checkpointer,
    state_schema=CustomState,
)

with open("hitl_agent.png", "wb") as f:
    f.write(agent.get_graph().draw_mermaid_png())


config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": "Let's go. What is the country of most orders?", "number_of_tool_calls": 0, "user_preferences": {"style": "formal"}}, config=config
)
print(response)

response = agent.invoke(Command(resume="approve"), config=config)
print(response)
