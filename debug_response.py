from openai import OpenAI
import time

API_KEY = "sk-__2SwruW7IZzzHm1pY9E7w"
BASE_URL = "https://aivie-xchange.sains.com.my/v1"
MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"

INPUT_TOKEN_COUNT = 1000
def generate_prompt(approx_tokens):
    num_chars = approx_tokens * 4
    text = "The quick brown fox jumps over the lazy dog. " * (num_chars // 45 + 1)
    return text[:num_chars]

prompt = generate_prompt(INPUT_TOKEN_COUNT)

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

print(f"Sending request to {BASE_URL} for model {MODEL_NAME}...")

try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        stream=True,
        temperature=0.7
    )
    
    print("Request sent, iterating response...")
    for chunk in response:
        print(f"Chunk: {chunk}")
        
except Exception as e:
    print(f"Caught Exception: {type(e).__name__}")
    print(f"Error: {e}")
    if hasattr(e, 'response'):
        print(f"Response status: {e.response.status_code}")
        print(f"Response text: {e.response.text}")
    if hasattr(e, 'body'):
        print(f"Body: {e.body}")

