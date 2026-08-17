from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import sqlite3
import requests
import os

load_dotenv()
ALPHA_VINTAGE_API_KEY = os.getenv("ALPHA_VINTAGE_API_KEY")
llm = ChatOpenAI(model="gpt-5-mini")

search_tool = DuckDuckGoSearchRun()

@tool
def get_stock_price(symbol:str) -> dict:
    """Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA)
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VINTAGE_API_KEY}"
    r = requests.get(url)
    return r.json()

tools = [search_tool, get_stock_price]
llm_with_tools = llm.bind_tools(tools)

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    msg = state["messages"]
    response = llm_with_tools.invoke(msg)
    
    return {"messages": [response]}

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")


checkpointer = SqliteSaver(conn=conn)

workflow = graph.compile(checkpointer=checkpointer)

def get_thread_ids():
    all_threads = set()
    
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    
    return all_threads

# response = workflow.invoke({"messages": [HumanMessage(content="How are you?")]}, {"configurable": {"thread_id": "thread-1"}})

# print(response["messages"][-1].content)
