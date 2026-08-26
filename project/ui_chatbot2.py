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

# Create model
model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.7,
)

st.set_page_config(
    page_title="Mistral AI Chatbot",
    page_icon="🤖",
)

st.title("🤖 Mistral AI Chatbot")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Choose AI Personality")

    personality = st.radio(
        "Select Mode",
        (
            "Professional AI",
            "Friendly AI",
            "Cynical AI",
        ),
    )

# Map personality to system prompt
if personality == "Professional AI":
    system_prompt = (
        "You are a Professional AI. "
        "You respond in a polished and precise manner."
    )

elif personality == "Friendly AI":
    system_prompt = (
        "You are a Friendly AI. "
        "You respond in a warm and chatty manner."
    )

else:
    system_prompt = (
        "You are a Cynical AI. "
        "You respond in a critical and sarcastic manner."
    )

# Initialize session state
if (
    "messages" not in st.session_state
    or st.session_state.get("current_personality") != personality
):
    st.session_state.current_personality = personality

    st.session_state.messages = [
        SystemMessage(content=system_prompt)
    ]

# Display chat history
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

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    response = model.invoke(st.session_state.messages)

    st.session_state.messages.append(
        AIMessage(content=response.content)
    )

    with st.chat_message("assistant"):
        st.markdown(response.content)
