from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool, Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.callbacks.tracers import LangChainTracer
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

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
    # pprint.pp(messages)
    response = llm_with_tools.invoke(messages, config=llm_config)
    pprint.pp(response)
    return {"messages": [response]}

def rewrite(state: State):
    print("---rewrite---")
    user_input = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)][-1].content
    prompt = PromptTemplate(
        template="""Given the user's original input, please try to reason about the underlying semantic intent / meaning, and formulate a more concise input.\n
        Origin user's input: {user_input}\n
        Improved user's input:""",
        input_variables=["user_input"]
    )
    chain = prompt | llm_with_tools
    response = chain.invoke({"user_input": user_input}, config=llm_config)
    pprint.pp(response)
    return {"messages": [response]}

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
# enable memory:
# graph = builder.compile(checkpointer=MemorySaver())

# run the agent.
while True:
    user_input = input("User: ")
    if user_input.lower() in ["q", "quit", "exit"]:
        print("Bye")
        break
    else:
        # for output in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config=llm_config, stream_mode=None):
        #     for key, value in output.items():
        #         pprint.pprint(f"Output from node '{key}':")
        #         pprint.pprint("---")
        #         pprint.pprint(value, indent=2, width=80, depth=None)
        #     pprint.pprint("---")
        graph.invoke({"messages": [{"role": "user", "content": user_input}]}, config=llm_config)