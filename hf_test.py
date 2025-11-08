import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ielasa .env failu
load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print("Token:", token[:10] + "..." if token else "NAV ATRASTS!")

# pieslēdzamies Hugging Face
client = InferenceClient(model="facebook/bart-large-cnn", token=token)

text = """Latvia is a country in Northern Europe on the Baltic Sea. 
Its capital and largest city is Riga. Latvia is part of the European Union."""

result = client.summarization(text)
print("\n=== KOPSAVILKUMS ===")
print(result)
