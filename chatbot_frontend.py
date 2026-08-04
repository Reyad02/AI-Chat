import streamlit as st
from chatbot_backend import workflow
from langchain_core.messages import HumanMessage
import uuid

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

if "message_history" not in st.session_state:
    st.session_state["message_history"]=[]

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
    
st.sidebar.title("AI-Chat")
st.sidebar.button("New Chat")

for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.text(msg["content"])
    
user_msg = st.chat_input("Write here...")

if user_msg:
    st.session_state["message_history"].append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.text(user_msg)
    
    with st.chat_message("assistant"):
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
    
    
