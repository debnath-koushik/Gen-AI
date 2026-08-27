import streamlit as st

from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

# ChatPromptTemplate is used to create structured chat prompts
# with roles like system, human, and AI.
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel
from typing import List, Optional

from langchain_core.output_parsers import PydanticOutputParser


# -----------------------------
# Pydantic Model
# -----------------------------

class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str


# -----------------------------
# Parser
# -----------------------------

parser = PydanticOutputParser(
    pydantic_object=Movie
)


# -----------------------------
# Load Environment Variables
# -----------------------------

load_dotenv()


# -----------------------------
# Mistral Model
# -----------------------------

model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.7,
)


# -----------------------------
# Prompt
# -----------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Extract movie information from the paragraph.

        {format_instructions}
        """
    ),
    (
        "human",
        """
        Here is the paragraph to analyze:

        {paragraph}
        """
    )
])


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="Movie Information Extractor",
    page_icon="🎬"
)

st.title("Movie Information Extractor")

paragraph = st.text_area(
    "Give your paragraph:",
    height=250
)


if st.button("Extract Information"):

    if paragraph.strip():

        final_prompt = prompt.invoke({
            "paragraph": paragraph,
            "format_instructions": parser.get_format_instructions()
        })

        response = model.invoke(final_prompt)

        # Convert LLM response into structured Pydantic object
        movie_data = parser.parse(response.content)

        # Store structured object in session state
        st.session_state.movie_data = movie_data

        st.subheader("Structured Output")

        # Display the complete Pydantic structure
        st.json(movie_data.model_dump())

    else:

        st.warning("Please enter a paragraph.")
