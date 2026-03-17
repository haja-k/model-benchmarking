import argparse
import asyncio
import json
import os
import time
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

DEFAULT_API_KEY = os.getenv("API_KEY")
DEFAULT_API_BASE = os.getenv("API_BASE_URL")
DEFAULT_MODEL = os.getenv("MODEL_SI_QWEN")
DEFAULT_TIMEOUT = float(os.getenv("DEFAULT_TIMEOUT", "60.0"))
DEFAULT_PROMPT_TOKENS = int(os.getenv("DEFAULT_PROMPT_TOKENS", "1000"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "512"))
DEFAULT_MIN_TOKENS = int(os.getenv("DEFAULT_MIN_TOKENS", "256"))


@dataclass
class TimeoutConfig:
    api_base: str
    api_key: str
    model: str
    timeout: float  # Client-side timeout in seconds
    prompt_tokens: int
    max_tokens: int
    min_tokens: int
    output_file: Optional[str] = None


@dataclass
class TimeoutResult:
    test_name: str
    success: bool
    timeout_occurred: bool
    partial_response: bool
    total_time: float
    ttft: float
    tokens_received: int
    tokens_expected: int
    error: Optional[str] = None
    response_text: Optional[str] = None


class TimeoutTester:
    def __init__(self, config: TimeoutConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

    def generate_prompt(self, target_tokens: int) -> str:
        base_text = "The quick brown fox jumps over the lazy dog. "
        chars_needed = target_tokens * 4
        repeats = (chars_needed // len(base_text)) + 1
        return (base_text * repeats)[:chars_needed]

    async def test_request(self, test_name: str, prompt: str, timeout_seconds: float) -> TimeoutResult:
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens,
            "stream": True,
            "temperature": 0.0,
            "stream_options": {"include_usage": True}
        }
        
        if self.config.min_tokens:
            payload["min_tokens"] = self.config.min_tokens

        start_time = time.perf_counter()
        first_token_time = None
        last_token_time = None
        token_count = 0
        response_text = ""
        timeout_occurred = False
        error_msg = None

        # Use the specified timeout
        try:
            url = f"{self.config.api_base}/chat/completions"
            if self.config.api_base.endswith("/"):
                url = f"{self.config.api_base}chat/completions"

            async with httpx.AsyncClient(
                headers=self.headers,
                timeout=httpx.Timeout(timeout_seconds, connect=10.0)
            ) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        error_msg = await response.aread()
                        return TimeoutResult(
                            test_name=test_name,
                            success=False,
                            timeout_occurred=False,
                            partial_response=False,
                            total_time=time.perf_counter() - start_time,
                            ttft=0,
                            tokens_received=0,
                            tokens_expected=self.config.max_tokens,
                            error=f"HTTP {response.status_code}: {error_msg.decode('utf-8')}"
                        )

                    async for line in response.aiter_lines():
                        if not line or line.strip() == "":
                            continue
                        
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                
                                if "usage" in data:
                                    token_count = data["usage"].get("completion_tokens", token_count)
                                
                                if len(data.get("choices", [])) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        last_token_time = time.perf_counter()
                                        response_text += content
                                        token_count += 1
                            except json.JSONDecodeError:
                                continue

        except httpx.TimeoutException:
            timeout_occurred = True
            error_msg = f"Client timeout after {timeout_seconds}s"
        except Exception as e:
            error_msg = str(e)

        end_time = time.perf_counter()
        
        ttft = (first_token_time - start_time) if first_token_time else 0
        
        return TimeoutResult(
            test_name=test_name,
            success=not timeout_occurred and error_msg is None,
            timeout_occurred=timeout_occurred,
            partial_response=timeout_occurred and token_count > 0,
            total_time=end_time - start_time,
            ttft=ttft,
            tokens_received=token_count,
            tokens_expected=self.config.max_tokens,
            error=error_msg,
            response_text=response_text[:500] if response_text else None
        )

    async def run_all_tests(self) -> List[TimeoutResult]:
        results = []
        
        # Test 1: Short timeout (should timeout)
        print("\n=== Test 1: Short timeout (5s) ===")
        prompt = self.generate_prompt(self.config.prompt_tokens)
        result = await self.test_request("short_timeout_5s", prompt, 5.0)
        results.append(result)
        self.print_result(result)

        # Test 2: Medium timeout (30s)
        print("\n=== Test 2: Medium timeout (30s) ===")
        result = await self.test_request("medium_timeout_30s", prompt, 30.0)
        results.append(result)
        self.print_result(result)

        # Test 3: Long timeout (120s)
        print("\n=== Test 3: Long timeout (120s) ===")
        result = await self.test_request("long_timeout_120s", prompt, 120.0)
        results.append(result)
        self.print_result(result)

        # Test 4: Very long prompt (high token count)
        print("\n=== Test 4: Large prompt (5000 tokens) ===")
        large_prompt = self.generate_prompt(5000)
        result = await self.test_request("large_prompt_5000tok", large_prompt, 60.0)
        results.append(result)
        self.print_result(result)

        # Test 5: Very large prompt (10000 tokens)
        print("\n=== Test 5: Very large prompt (10000 tokens) ===")
        very_large_prompt = self.generate_prompt(10000)
        result = await self.test_request("very_large_prompt_10000tok", very_large_prompt, 120.0)
        results.append(result)
        self.print_result(result)

        # Test 6: Large output (high max_tokens)
        print("\n=== Test 6: Large output (2048 tokens) ===")
        large_output_config = TimeoutConfig(
            api_base=self.config.api_base,
            api_key=self.config.api_key,
            model=self.config.model,
            timeout=180.0,
            prompt_tokens=self.config.prompt_tokens,
            max_tokens=2048,
            min_tokens=1000,
            output_file=None
        )
        tester = TimeoutTester(large_output_config)
        result = await tester.test_request("large_output_2048tok", prompt, 180.0)
        results.append(result)
        self.print_result(result)

        # Test 7: Sequential requests (simulating real client behavior)
        print("\n=== Test 7: Sequential requests (5 requests) ===")
        for i in range(5):
            result = await self.test_request(f"sequential_req_{i+1}", prompt, 60.0)
            results.append(result)
            self.print_result(result)
            await asyncio.sleep(1)  # 1s between requests

        # Test 8: Concurrent requests (burst)
        print("\n=== Test 8: Concurrent requests (10 parallel) ===")
        tasks = [
            self.test_request(f"concurrent_req_{i+1}", prompt, 60.0)
            for i in range(10)
        ]
        concurrent_results = await asyncio.gather(*tasks)
        for result in concurrent_results:
            results.append(result)
            self.print_result(result)

        return results

    def print_result(self, result: TimeoutResult):
        status = "✓ SUCCESS" if result.success else "✗ FAILED"
        timeout_mark = " [TIMEOUT]" if result.timeout_occurred else ""
        partial_mark = " [PARTIAL]" if result.partial_response else ""
        
        print(f"  {status}{timeout_mark}{partial_mark}")
        print(f"  Time: {result.total_time:.2f}s | TTFT: {result.ttft*1000:.0f}ms")
        print(f"  Tokens: {result.tokens_received}/{result.tokens_expected}")
        if result.error:
            print(f"  Error: {result.error}")
        if result.partial_response and result.response_text:
            print(f"  Partial response: {result.response_text[:100]}...")
        print()

    def save_results(self, results: List[TimeoutResult]):
        if not self.config.output_file:
            return
        
        data = {
            "config": {
                "api_base": self.config.api_base,
                "model": self.config.model,
                "timeout": self.config.timeout,
                "prompt_tokens": self.config.prompt_tokens,
                "max_tokens": self.config.max_tokens,
                "min_tokens": self.config.min_tokens
            },
            "results": [asdict(r) for r in results]
        }
        
        with open(self.config.output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {self.config.output_file}")


async def main():
    parser = argparse.ArgumentParser(description="Timeout Diagnostic Tool")
    parser.add_argument("--base-url", default=DEFAULT_API_BASE, help="API Base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API Key (default: from .env)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Default timeout in seconds")
    parser.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS, help="Prompt token count")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens to generate")
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS, help="Min tokens to generate")
    parser.add_argument("--output", default="timeout_test_results.json", help="Output JSON file")
    parser.add_argument("--single", action="store_true", help="Run single quick test only (useful for quick diagnostics)")
    
    args = parser.parse_args()
    
    config = TimeoutConfig(
        api_base=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
        prompt_tokens=args.prompt_tokens,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        output_file=args.output
    )
    
    print("=" * 60)
    print("TIMEOUT DIAGNOSTIC TOOL")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"API: {args.base_url}")
    print(f"Default timeout: {args.timeout}s")
    print("=" * 60)
    
    tester = TimeoutTester(config)
    
    if args.single:
        print("\n=== SINGLE QUICK TEST ===")
        prompt = tester.generate_prompt(config.prompt_tokens)
        result = await tester.test_request("single_test", prompt, config.timeout)
        tester.print_result(result)
        results = [result]
    else:
        results = await tester.run_all_tests()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    timeouts = [r for r in results if r.timeout_occurred]
    successes = [r for r in results if r.success]
    partials = [r for r in results if r.partial_response]
    
    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successes)}")
    print(f"Timeouts: {len(timeouts)}")
    print(f"Partial responses (got some tokens before timeout): {len(partials)}")
    
    if timeouts:
        print("\nTimeout scenarios:")
        for t in timeouts:
            print(f"  - {t.test_name}: {t.error}")
    
    if partials:
        print("\nPartial response scenarios (may indicate slow generation):")
        for p in partials:
            print(f"  - {p.test_name}: {p.tokens_received}/{p.tokens_expected} tokens in {p.total_time:.2f}s")
    
    tester.save_results(results)


if __name__ == "__main__":
    asyncio.run(main())