from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

# load .env variables
load_dotenv()


# Mistral model using chat_model
model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.7,
)

# single query
# prompt = input("User: ")
# response = model.invoke(prompt)

# print("AI: ", response.content)

# multiple query
# print("----------------- Welcome User!, Press 0 to exit -----------------")
# while True:
#     prompt = input("User: ")
    
#     if prompt == "0":
#         break
    
#     response = model.invoke(prompt)
#     print("AI: ", response.content)

# multiple query with stored history
"""
* Problems with our current short-term memory
- No role separation (no system / user / assistant distinction)
- Just raw string -> weak conversation structure
- Memory keeps growing infinitely
- Will hit token limit
- API cost increases over time
- Slower response as history grows
- No trimming machanism
- No summarization of old chats
- Not production scable
- No control over context window
"""

# messages = []

# print("----------------- Welcome User!, Press 0 to exit -----------------")
# while True:
#     prompt = input("User: ")

#     if prompt == "0":
#         break
    
#     messages.append(prompt)

#     response = model.invoke(messages)
#     messages.append(response.content)
#     print("AI: ", response.content)

"""

"""
# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# messages = [
#     SystemMessage(content="You are a funny AI agent")
# ]

# print("----------------- Welcome User!, Press 0 to exit -----------------")
# while True:
#     prompt = input("User: ")

#     if prompt == "0":
#         break

#     messages.append(HumanMessage(content=prompt))

#     response = model.invoke(messages)
#     messages.append(AIMessage(content=response.content))
#     print("AI: ", response.content)

# print(messages)

# Choose AI personalization
print("Choose your AI mode")
print("Choose 1 for Professional AI")
print("Choose 2 for Friendly AI")
print("Choose 3 for Cynical AI")
choice = input("tell me your coice -")

if choice == "1":
    mode = "You are a Professional AI. You responed with polished and precise"
elif choice == "2":
    mode = "You are a Friendly AI. You responed with warm and chatty"
else:
    mode = "You are a Cynical AI. You responed with critical and sarcastic"
    
messages = [
    SystemMessage(content=mode)
]

print("----------------- Welcome User!, Press 0 to exit -----------------")
while True:
    prompt = input("User: ")

    if prompt == "0":
        break

    messages.append(HumanMessage(content=prompt))

    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("AI: ", response.content)

print(messages)
