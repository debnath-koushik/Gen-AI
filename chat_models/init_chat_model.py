from dotenv import load_dotenv
"""
init_chat_model LangChain ka universal initializer hai. Jab model kabhi bhi change ho sakta ho tab hum iska use karte hein.
"""
from langchain.chat_models import init_chat_model

# load .env variables
load_dotenv()

# HuggingFace model using init_chat_model
model = init_chat_model(
    "Qwen/Qwen3-4B",
    model_provider="huggingface",
)

# Mistral model using init_chat_model
# _mistral_model = init_chat_model(
#     model="mistral-small-latest",
#     model_provider="mistralai"
# )

response = model.invoke("What is the national animal of India?")

print(response.content)