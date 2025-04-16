from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool, Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.callbacks.tracers import LangChainTracer

from typing import Annotated
from typing_extensions import TypedDict

import pprint
import os

os.environ["SERPER_API_KEY"] = "<google serper api key>"
os.environ["LANGSMITH_TRACING"] = "True"
os.environ["LANGSMITH_ENDPOINT"] = "<langsmith endpoint>"
os.environ["LANGSMITH_API_KEY"] = "<langsmith api key>"
os.environ["LANGSMITH_PROJECT"] = "<langsmith project>"
tracer = LangChainTracer()
llm_config = {
    "thread_id": "1",
    "callbacks": [tracer]
}

# define the tools available to the llm:
@tool
def calculator(a: float, b: float, op: str) -> float:
    """
    A simple calculator function that performs basic arithmetic operations.

    Args:
        a (float): The first number in the operation.
        b (float): The second number in the operation.
        op (str): The operation to perform. Supported operations are:
            - "+" for addition
            - "-" for subtraction
            - "*" for multiplication
            - "/" for division
            - "%" for modulus

    Returns:
        float: The result of the arithmetic operation.

    Raises:
        ValueError: If an unsupported operator is provided.
    """
    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        return a / b
    elif op == "%":
        return a % b
    else:
        raise(f"Unsupported operator: {op}")

searcher = GoogleSerperAPIWrapper()
llm = ChatOllama(model="qwen2.5")
tools = [
    calculator,
    Tool(
        name="internet-searcher",
        func=searcher.run,
        description="This tool can search the internet for any real-time data"
    )
]
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

# define the nodes and edges in the graph:
def process(state: State):
    print("---process---")
    messages = state["messages"]
    pprint.pp(messages)
    response = llm_with_tools.invoke(messages, config=llm_config)
    pprint.pp(response)
    return {"messages": [response]}

def rewrite(state: State):
    print("---rewrite---")
    return {}

def need_call_tools(state: State):
    print("---need_call_tools---")
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"
    else:
        print("No tool call available")
        return "__end__"

def is_output_good_enough(state: State):
    print("---is_output_good_enough---")
    tool_message = state["messages"][-1]
    pprint.pp(tool_message)
    rate = input("How do you like the above answer? If it does not satisfy your needs, I can regenerate the answer (Y/N): ")
    if rate.lower() == 'y':
        return "__end__"
    else:
        print("regenerating the answer...")
        return "rewrite"

# define the graph structure:
builder = StateGraph(State)
builder.add_node("process", process)
builder.add_node("tools", ToolNode(tools))
builder.add_node("rewrite", rewrite)
builder.add_edge(START, "process")
builder.add_conditional_edges("process", need_call_tools)
builder.add_conditional_edges("tools", is_output_good_enough)
builder.add_edge("rewrite", "process")
graph = builder.compile()

graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Whats the weather like today in Suzhou?"
            }
        ]
    },
    config=llm_config
)