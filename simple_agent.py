import operator
from typing import Annotated, Callable
from langgraph.graph import StateGraph, END, START, MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.messages import ToolMessage


@tool
def search_database(query: str):
    """Searches the database"""
    return f"Results for query: Honduras"


def get_llm(model_name: str):
    if model_name == "gemini-2.0-flash-lite":
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
    elif model_name == "gpt-4o-mini":
        return ChatOpenAI(model=model_name, temperature=0.0)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def save_graph_png(filename: str, graph: StateGraph):
    graph_data = graph.get_graph().draw_mermaid_png()
    with open(filename, "wb") as f:
        f.write(graph_data)


class AgentState(MessagesState):
    number_of_calls: Annotated[int, operator.add]


workflow = StateGraph(AgentState)


tools = [search_database]
tools_by_name = {tool.name: tool for tool in tools}

class LlmNode:

    def __init__(self, llm, tools):
        self.runnable = llm.bind_tools(tools)
    
    def __call__(self, state: AgentState):
        response = self.runnable.invoke(state['messages'])
        return {"messages": [response], "number_of_calls": state.get("number_of_calls", 0) + 1}
        

llm_node = LlmNode(get_llm("gpt-4o-mini"), tools)


def route_to_tools(state: AgentState):
    if state['messages'][-1].tool_calls:
        return "tools"
    return END


def call_tools(state: AgentState):
    tool_messages = []
    for tool_call in state['messages'][-1].tool_calls:
        tool_response = tools_by_name[tool_call['name']].invoke(tool_call['args'])
        tool_messages.append(ToolMessage(content=tool_response, tool_call_id=tool_call["id"], name=tool_call['name']))
    return {"messages": tool_messages}
    #return {"messages": [ToolMessage("answer to my question is Honduras", tool_call_id=tool_call["id"])]}


workflow.add_node("call_llm", llm_node)
#workflow.add_node("tools", ToolNode(tools=tools))
workflow.add_node("tools", call_tools)

workflow.add_edge(START, "call_llm")
workflow.add_edge("tools", "call_llm")
#workflow.add_conditional_edges("call_llm", tools_condition)
workflow.add_conditional_edges("call_llm", route_to_tools, {"tools": "tools", END: END})

agent = workflow.compile()

save_graph_png("simple_agent.png", agent)

response = agent.invoke({"messages": "What country has the most orders in the db?"})
print(response)    


