import time
import os
from openai import OpenAI

# Configuration
# Using credentials from call_vllm.py
API_KEY = "sk-__2SwruW7IZzzHm1pY9E7w"
BASE_URL = "https://aivie-xchange.sains.com.my/v1"

# List of models to benchmark. 
# Please update this list with the actual model names you want to test.
MODELS = [
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "deepseek-ai/DeepSeek-V3.1",
    "openai/gpt-oss-120b"
]

# Benchmark settings
INPUT_TOKEN_COUNT = 1000  # Constant 1K input tokens
# The user mentioned 32K context window, but for the benchmark we use a fixed input.
# We will generate enough output tokens to measure speed accurately.
MAX_OUTPUT_TOKENS = 1024   

def generate_prompt(approx_tokens):
    """
    Generates a text prompt of approximately `approx_tokens` tokens.
    Approximation: 1 token ~= 4 characters.
    """
    num_chars = approx_tokens * 4
    # A simple repeated sentence to fill the context
    text = "The quick brown fox jumps over the lazy dog. " * (num_chars // 45 + 1)
    return text[:num_chars]

def benchmark_model(client, model_name):
    print(f"--------------------------------------------------")
    print(f"Benchmarking Model: {model_name}")
    
    prompt = generate_prompt(INPUT_TOKEN_COUNT)
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    start_time = time.time()
    first_token_time = None
    token_count = 0
    
    try:
        # Send streaming request
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
            temperature=0.7
        )
        
        print("Request sent, waiting for response...")
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.time()
                    print("First token received.")
                token_count += 1
                # Optional: print a dot every 10 tokens to show progress
                if token_count % 50 == 0:
                    print(".", end="", flush=True)
        
        print("\nGeneration complete.")
        end_time = time.time()
        
        if first_token_time is None:
            print(f"Error: No tokens received for {model_name}")
            return

        # Metrics Calculation
        ttft = first_token_time - start_time
        
        # Output Speed (Tokens Per Second)
        # We calculate based on the generation phase (after first token)
        generation_time = end_time - first_token_time
        
        # Avoid division by zero
        if generation_time > 0:
            tps = (token_count - 1) / generation_time
        else:
            tps = 0
            
        results = (
            f"Model: {model_name}\n"
            f"Input Tokens (approx): {INPUT_TOKEN_COUNT}\n"
            f"Output Tokens: {token_count}\n"
            f"Time to First Token (TTFT): {ttft:.4f} seconds\n"
            f"Output Speed (TPS): {tps:.2f} tokens/second\n"
            f"Total Duration: {end_time - start_time:.4f} seconds\n"
        )
        
        print("\nResults:")
        print(results)
        
        # Save to file
        # Sanitize model name for filename
        safe_model_name = model_name.replace("/", "_").replace(":", "_")
        filename = f"{safe_model_name}_benchmark.txt"
        
        with open(filename, "w") as f:
            f.write(results)
        print(f"Results saved to {filename}")
        
    except Exception as e:
        print(f"Failed to benchmark {model_name}: {e}")

def main():
    print(f"Initializing client with base_url={BASE_URL}")
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    
    if not MODELS:
        print("No models defined in the MODELS list. Please edit the script to add your models.")
        return

    for model in MODELS:
        benchmark_model(client, model)

if __name__ == "__main__":
    main()
