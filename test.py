from huggingface_hub import whoami
import os
from dotenv import load_dotenv

load_dotenv()

print(whoami(token=os.getenv("HUGGINGFACEHUB_API_TOKEN")))
