import json
import random
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
import os

load_dotenv(override=True)
# Attempt to use an env variable first, if unavailble fallback to the default localhost and port
LOCAL_MODEL_API_ENDPOINT = os.getenv("LOCAL_LLM") or os.getenv("WINDOWS_OLLAMA_IP") or "http://localhost:11434"


llm = OllamaLLM(
    model="llama3.1:8b",
    base_url=LOCAL_MODEL_API_ENDPOINT,
    temperature=0.2
)


with open('../data/processed/val.jsonl', 'r') as f:
    val_data = [json.loads(line) for line in f]

print(f"Loaded {len(val_data)} validation samples")

# Select a few random samples for testing
test_samples = random.sample(val_data, 3)

print("\n" + "="*80)
print("BASELINE MODEL TEST - Llama 3.1 8B (Pre-Fine-Tuning)")
print("="*80)

for idx, sample in enumerate(test_samples):
    print(f"\n --- Test Case {idx+1} ---")
    print(f"PATIENT INPUT:\n{sample['input'][:300]}...\n")

    # Create prompt
    prompt = f"""
    You are a medical triage assitant.  Based on the patient information below, provide:
    1. The likely medical specialty needed
    2. A clinical assessment
    

    {sample['input']}
    """

response = llm.invoke(prompt)

# print(response)

print(f"BASE MODEL OUTPUT:\n{response}\n")
print(f"EXPECTED OUTPUT:\n{sample['output'][:400]}...\n")
print("="*80)