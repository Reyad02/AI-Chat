import streamlit as st
from chatbot_backend import workflow, get_thread_ids
from langchain_core.messages import HumanMessage
import uuid

# ========== Utility Func ===========

def generate_thread_id():
    thread_id = uuid.uuid4()
    return str(thread_id)

def reset_state():
    thread_id=generate_thread_id()
    st.session_state["thread_id"]=thread_id
    st.session_state["message_history"]=[]
    st.session_state["threads"].add(thread_id)

# ========== State Var ===========

if "message_history" not in st.session_state:
    st.session_state["message_history"]=[]

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
    
if "threads" not in st.session_state:
    st.session_state["threads"] = get_thread_ids()

if st.session_state["thread_id"] not in st.session_state["threads"]:
    st.session_state["threads"].add(st.session_state["thread_id"])

    
st.sidebar.title("AI-Chat")
if st.sidebar.button("New Chat"):
    reset_state()

# ========== Showing all the thread ==========    
for thread in st.session_state["threads"]:
    
# ========== Select a specific thread ==========
    if st.sidebar.button(str(thread)):
        st.session_state["thread_id"] = thread
        history = workflow.get_state(config={"configurable": {"thread_id": thread}})
        messages = history.values["messages"]
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
        
# ========== Generator Function for streaming ==========        
        def response_streaming():
            for chunk in workflow.stream(
                {"messages": [HumanMessage(content=user_msg)]}, {"configurable": {"thread_id": st.session_state["thread_id"]}},
                stream_mode="messages",
                version="v2"
            ):
                if chunk["type"]=="messages":
                    token, metadata = chunk["data"]
                    
                    if token.content:
                        yield token.content
            
        response = st.write_stream(response_streaming())
    st.session_state["message_history"].append({"role": "assistant", "content": response})  
    
    
