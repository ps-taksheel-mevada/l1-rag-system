import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from commands.init_db import vector_store
from commands.agent import agent_executor


st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🦜")
st.title("Agentic RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


user_question = st.chat_input("How are you?")

if user_question:

    with st.chat_message("user"):
        st.markdown(user_question)

        st.session_state.messages.append(HumanMessage(user_question))

    result = agent_executor.invoke({"input": user_question, "chat_history":st.session_state.messages})
    

    ai_message = result["output"]

    with st.chat_message("assistant"):
        st.markdown(ai_message)

        st.session_state.messages.append(AIMessage(ai_message))
        print(st.session_state.messages)