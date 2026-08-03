from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
from dotenv import load_dotenv

# used this code to change default download location for store hugging face model data
os.environ['HF_HOME'] = 'D:/Koushik/huggingface_cache'

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)

chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("What is 2 + 2?")

print(response.content)