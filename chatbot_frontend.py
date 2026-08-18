import streamlit as st
from chatbot_backend import workflow, get_thread_ids, ingest_pdf, thread_documents_metadata
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# ========== Utility Func ===========

def generate_thread_id():
    thread_id = uuid.uuid4()
    return str(thread_id)

def reset_state():
    thread_id=generate_thread_id()
    st.session_state["thread_id"]=thread_id
    st.session_state["message_history"]=[]
    add_thread(thread_id)

def add_thread(thread_id):
    if thread_id not in st.session_state["threads"]:
        st.session_state["threads"].add(thread_id)
        
def load_conversation(thread_id):
    state = workflow.get_state(config={"configurable": {"thread_id": thread}})
    return state.values.get("messages", [])

# ========== State Var ===========

if "message_history" not in st.session_state:
    st.session_state["message_history"]=[]

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
    
if "threads" not in st.session_state:
    st.session_state["threads"] = get_thread_ids()

if st.session_state["thread_id"] not in st.session_state["threads"]:
    add_thread(st.session_state["thread_id"])
    
if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}
    
# The ingested_docs will be look like the following output
# {
#     "thread_id_1": {
#         "financial_report.pdf": {
#             "filename": "financial_report.pdf",
#             "documents": 25,
#             "chunks": 120
#         },
#         "balance_sheet.pdf": {
#             "filename": "balance_sheet.pdf",
#             "documents": 10,
#             "chunks": 45
#         }
#     }
# }  

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
# setdeafult first to search using the thread_key. if it finds it return the value if can't find then set the thread_key and the value of thread key is like {}
    
st.sidebar.title("AI-Chat")
if st.sidebar.button("New Chat"):
    reset_state()
    st.rerun()
    
if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No pdf indexed yet...")
    
upload_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if upload_pdf:
    if upload_pdf.name in thread_docs:
        st.sidebar.info(f"`{upload_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                file_bytes=upload_pdf.getvalue(),
                thread_id=thread_key,
                filename=upload_pdf.name
            )
            thread_docs[upload_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

            

# ========== Showing all the thread ==========    
for thread in st.session_state["threads"]:
    
# ========== Select a specific thread ==========
    if st.sidebar.button(str(thread)):
        st.session_state["thread_id"] = thread
        messages = load_conversation(st.session_state["thread_id"])
        temp_messages = []
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                temp_messages.append({"role": "user", "content": msg.content})
            else: 
                temp_messages.append({"role": "assistant", "content": msg.content})
        
        st.session_state["message_history"] = temp_messages

# ========= Showing the full chat ========== 
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.text(msg["content"])
        
# ========== Taking user input ========== 
user_msg = st.chat_input("Write here...")

if user_msg:
    st.session_state["message_history"].append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.text(user_msg)
    
    with st.chat_message("assistant"):
        status_holder = {"box": None}
        
# ========== Generator Function for streaming ==========        
        def response_streaming():
            for chunk in workflow.stream(
                {"messages": [HumanMessage(content=user_msg)]}, {"configurable": {"thread_id": st.session_state["thread_id"]}},
                stream_mode="messages",
                version="v2"
            ):
                if chunk["type"]=="messages":
                    token, metadata = chunk["data"]
                    
                    if isinstance(token, ToolMessage):
                        tool_name = getattr(token, "name", "tool")
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(
                                label=f"Using {tool_name} ...", expanded=True
                            )
                        else:
                            status_holder["box"].update(
                                label=f"Using {tool_name}",
                                state="running",
                                expanded=True
                            )
                    
                    if token.content and isinstance(token, AIMessage):
                        yield token.content
            
        response = st.write_stream(response_streaming())
        
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="Tool Finished", state="complete", expanded=False
            )
    st.session_state["message_history"].append({"role": "assistant", "content": response})  
    
    doc_meta = thread_documents_metadata(thread_id=thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )
        
# st.divider
