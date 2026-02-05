from typing import List, Callable

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI


@tool
def add_numbers(number_a: float, number_b: float) -> float:
    """
    Adds two float numbers and returns the result as a float.
    """
    return number_a + number_b


@tool
def clarification_rephraser(last_user_message: str, last_ai_response: str) -> str:
    """Use this tool whne a user asked for a clarification of the last AI message.
    """
    clarification_prompt = [
        SystemMessage("You are a communication and rephrasing expert. Rephrase the last AI message to clarify the users confusion."),
        HumanMessage(f"AI said: {last_ai_response}. User asked: {last_user_message}. Provide a clearer rephrased AI message.")
        ]
    llm = get_llm()
    response = llm.invoke(clarification_prompt)
    print(f"Clarifiaction RESPONSE: {response.content}")
    return response.content


my_tools = [add_numbers, clarification_rephraser]


def get_llm(llm_name: str = "gemini-2.5-flash", tools: List[Callable] = None) -> ChatGoogleGenerativeAI:
    llm = ChatGoogleGenerativeAI(model=llm_name, temperature=0.0)
    if tools:
        llm = llm.bind_tools(tools)
    return llm


if __name__ == "__main__":
    llm = get_llm(tools=my_tools)
    
    #prompt = "What is the sum 2 and 5?"
    prompt = [AIMessage("What are the main metrics you should consider for scaling your load based scaling on EKS?"), HumanMessage("I don't understand what EKS means?")]

    response = llm.invoke(prompt)
    print(response)

    tc = response.tool_calls[0]
    print(clarification_rephraser.invoke(tc))
    # print(add_numbers.invoke(tc))
    

