import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_mistralai import ChatMistralAI

# Load environment variables
load_dotenv()

# Create Mistral model
model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.7,
)

st.set_page_config(page_title="Mistral Chatbot", page_icon="🤖")

st.title("🤖 Mistral AI Chatbot")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a funny AI agent")
    ]

# Display previous chat messages (skip system message)
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Chat input
prompt = st.chat_input("Type your message...")

if prompt:
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to history
    st.session_state.messages.append(HumanMessage(content=prompt))

    # Get AI response
    response = model.invoke(st.session_state.messages)

    # Save AI response
    st.session_state.messages.append(
        AIMessage(content=response.content)
    )

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(response.content)
