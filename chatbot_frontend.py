import streamlit as st
from chatbot_backend import workflow
from langchain_core.messages import HumanMessage

if "message_history" not in st.session_state:
    st.session_state["message_history"]=[]

for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.text(msg["content"])
    
    
user_msg = st.chat_input("Write here...")

if user_msg:
    st.session_state["message_history"].append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.text(user_msg)
        
    response = workflow.invoke({"messages": [HumanMessage(content=user_msg)]}, {"configurable": {"thread_id": "thread-1"}})
    st.session_state["message_history"].append({"role": "assistant", "content": response["messages"][-1].content})
    with st.chat_message("assistant"):
            st.text(response["messages"][-1].content)
    
    
