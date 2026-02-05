from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)


class AgentState(MessagesState):
    number_of_steps: int


def call_model(state: AgentState) -> dict:
    print(state)
    response = model.invoke(state['messages'])
    return {"messages": [response], "number_of_steps": state.get("number_of_steps", 0) + 1}


def should_end(state: AgentState):
    if state['number_of_steps'] >= 2:
        return "end"
    return "continue"


workflow = StateGraph(AgentState)

workflow.add_node("llm", call_model)

workflow.add_edge(START, "llm")
#workflow.add_edge("llm", END)
workflow.add_conditional_edges("llm", should_end, {"end": END, "continue": "llm"})

agent = workflow.compile()


image_data = agent.get_graph().draw_mermaid_png()
with open("custom_agent.png", "wb") as f:
    f.write(image_data)


response = agent.invoke({"messages": [HumanMessage("What is going on?")]})
print(response)
