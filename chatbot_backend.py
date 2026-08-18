from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Dict, Optional, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import sqlite3
import requests
import os
import tempfile

load_dotenv()
ALPHA_VINTAGE_API_KEY = os.getenv("ALPHA_VINTAGE_API_KEY")
llm = ChatOpenAI(model="gpt-5-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

_THREAD_RETRIEVERS: Dict[str, Any] = {}
# _THREAD_RETRIEVERS = {
#     "thread-A": retriever_for_company_report,
#     "thread-B": retriever_for_research_paper,
#     "thread-C": retriever_for_financial_report
# }

_THREAD_METADATA: Dict[str, dict] = {}
# _THREAD_METADATA = {
#     "thread-A": {
#         "filename": "company_report.pdf",
#         "documents": 50,
#         "chunks": 320
#     },
#     "thread-B": {
#         "filename": "research.pdf",
#         "documents": 20,
#         "chunks": 145
#     }
# }

def _get_Retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[str(thread_id)]

    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str]=None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    
    if not file_bytes:
        raise ValueError("No bytes received")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000, chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)
        
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type= "similarity", search_kwargs={"k": 4}
        )
        
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks)
        }
        
        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks)            
        }
    finally:
        try:
            os.remove(temp_path)
        
        except OSError:
            pass
            
search_tool = DuckDuckGoSearchRun()

@tool
def get_stock_price(symbol:str) -> dict:
    """Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA)
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VINTAGE_API_KEY}"
    r = requests.get(url)
    return r.json()

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    
    retriever = _get_Retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for the this chat. Upload a pdf first",
            "query": query
        }
    
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]
    
    return{
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename")
    }

tools = [search_tool, get_stock_price, rag_tool]
llm_with_tools = llm.bind_tools(tools)

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState, config = None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
        
    system_message = SystemMessage(
        content=f"""You are a helpful assistant. For questions about the uploaded PDF, call the `rag_tool` and include the thread_id {thread_id}. 
        
        You can also use the web search, stock price, and calculator tools when helpful. If no document is available, ask the user to upload a PDF."""
    )
    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
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

def thread_documents_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})

# response = workflow.invoke({"messages": [HumanMessage(content="How are you?")]}, {"configurable": {"thread_id": "thread-1"}})

# print(response["messages"][-1].content)
