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

st.title("🎬 Movie Information Extractor")

st.write("Enter a movie paragraph to extract structured information.")


paragraph = st.text_area(
    "Movie Paragraph",
    placeholder="Enter your movie paragraph here...",
    height=250
)


if st.button("Extract Information"):

    if paragraph.strip():

        final_prompt = prompt.invoke({
            "paragraph": paragraph,
            "format_instructions": parser.get_format_instructions()
        })

        response = model.invoke(final_prompt)

        st.subheader("Extracted Information")

        st.write(response.content)

    else:

        st.warning("Please enter a paragraph.")
