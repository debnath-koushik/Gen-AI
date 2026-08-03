from dotenv import load_dotenv

# load .env variables
load_dotenv()

"""
Model class LangChain ka unique initializer hai. Jab koi unique model ke upar hum ek project bnana chahete hein tab hum iska use karte hein.
"""


# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# HuggingFace model using chat_model
# llm = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
# )

# model = ChatHuggingFace(llm=llm)

from langchain_mistralai import ChatMistralAI

# Mistral model using chat_model
model = ChatMistralAI(
    model="mistral-small-latest", # mistral-small-2506
    temperature=0.7,
    max_tokens=1024,
)

response = model.invoke("What is the national animal of India?")

print(response.content)
