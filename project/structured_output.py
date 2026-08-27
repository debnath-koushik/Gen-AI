from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

# ChatPromptTemplate is used to create structured chat prompts with roles like system, human, and AI.
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)

load_dotenv()

# Mistral model using chat_model
model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.7,
)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
     Extract movie information from the paragraph
     {format_instrucions}
     """),
    (
        "human",
        """
        Here is the paragraph to analyze:
        {paragraph}
        """
    )
])

paragraph = input("Give your paragraph: ")

final_prompt = prompt.invoke({
    "paragraph": paragraph,
    "format_instrucions": parser.get_format_instructions()
})

response = model.invoke(final_prompt)
movie_data = parser.parse(response.content)

# print(response.content)
print(movie_data)
