from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

# load .env variables
load_dotenv()


# Mistral model using chat_model
model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.7,
)

prompt = input("User: ")
response = model.invoke(prompt)

print("AI: ", response.content)