from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
# PromptTemplate is used to create simple text-based prompts with dynamic variables.
# ChatPromptTemplate is used to create structured chat prompts with roles like system, human, and AI.
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

load_dotenv()

# Mistral model using chat_model
model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.7,
)

# prompt = PromptTemplate.from_template("Say {foo}")
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
     You are an information extraction and summarization assistant.

    Analyze the following paragraph and extract all useful and explicitly mentioned
    information from it.

    IMPORTANT RULES:
    1. Use ONLY information present in the provided paragraph.
    2. Do NOT invent, assume, or add information from your own knowledge.
    3. If a field is not mentioned, write "Not mentioned".
    4. Keep extracted information concise and accurate.
    5. For cast, list all people mentioned in the paragraph.
    6. Provide a short summary of the paragraph in 2-3 sentences.
    7. Identify the main topics or themes mentioned in the paragraph.

    Return the result using exactly this structure:

    Movie Information:
    - Movie Name:
    - Release Year:
    - Director:
    - Genre:
    - Cast:
    - Main Characters:
    - Plot:
    - Themes:
    - Setting:
    - Notable Features:
    - Music / Composer:
    - Visual Style / Cinematography:
    - Keywords:

    Quick Summary:
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
    "paragraph": paragraph
})

response = model.invoke(final_prompt)

print(response.content)
