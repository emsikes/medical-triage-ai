from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
import os
import json

load_dotenv(override=True)
# Attempt to use an env variable first, if unavailble fallback to the default localhost and port
LOCAL_MODEL_API_ENDPOINT = os.getenv("LOCAL_LLM") or os.getenv("WINDOWS_OLLAMA_IP") or "http://localhost:11434"

llm = OllamaLLM(
    model="llama3.1:8b",
    base_url=LOCAL_MODEL_API_ENDPOINT,
    temperature=0.2
)

prompt = """
What are 5 steps to advance as an AI Engineer?
"""

response = llm.invoke(prompt)
print(response)


