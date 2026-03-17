import argparse
import asyncio
import json
import logging
import os
import time
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import httpx
from dotenv import load_dotenv

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

load_dotenv()

DEFAULT_API_KEY = os.getenv("API_KEY")
DEFAULT_BASE_URL = os.getenv("API_BASE_URL")
DEFAULT_MODEL = os.getenv("MODEL")


@dataclass
class BenchmarkConfig:
    api_base: str
    api_key: str
    model: str
    mode: str
    num_runs: int
    warmup_runs: int
    concurrency: int
    max_tokens: int
    temperature: float
    prompt_tokens: int
    min_tokens: Optional[int] = None
    duration: int = 0
    output_file: Optional[str] = None


@dataclass
class RequestMetrics:
    request_id: str
    start_time: float
    ttft: float
    total_time: float
    generation_time: float
    output_tokens: int
    input_tokens: int
    throughput: float
    status: str = "success"
    error: Optional[str] = None


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        self.config.model = self.config.model.strip()
        self.timeout = httpx.Timeout(120.0, connect=10.0)

    def generate_prompt(self, target_tokens: int) -> str:
        base_text = "The quick brown fox jumps over the lazy dog. "
        chars_needed = target_tokens * 4
        repeats = (chars_needed // len(base_text)) + 1
        return (base_text * repeats)[:chars_needed]

    async def run_single_request(self, client: httpx.AsyncClient, req_id: str, prompt: str) -> RequestMetrics:
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens,
            "stream": True,
            "temperature": self.config.temperature,
            "stream_options": {"include_usage": True}
        }
        
        if self.config.min_tokens:
            payload["min_tokens"] = self.config.min_tokens

        start_time = time.perf_counter()
        first_token_time = None
        last_token_time = None
        token_count = 0
        
        try:
            url = f"{self.config.api_base}/chat/completions"
            if self.config.api_base.endswith("/"):
                 url = f"{self.config.api_base}chat/completions"

            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    error_msg = await response.aread()
                    logger.error(f"Request {req_id} failed: {response.status_code} - {error_msg.decode('utf-8')}")
                    return RequestMetrics(
                        request_id=str(req_id),
                        start_time=start_time,
                        ttft=0, total_time=0, generation_time=0, output_tokens=0, input_tokens=0, throughput=0,
                        status="failed", error=str(response.status_code)
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
                                usage = data["usage"]
                                token_count = usage.get("completion_tokens", token_count)
                                input_tokens = usage.get("prompt_tokens", self.config.prompt_tokens)
                            
                            if len(data.get("choices", [])) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    last_token_time = time.perf_counter()
                                    token_count += 1
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Request {req_id} exception: {e}")
            return RequestMetrics(
                request_id=str(req_id),
                start_time=start_time,
                ttft=0, total_time=0, generation_time=0, output_tokens=0, input_tokens=0, throughput=0,
                status="error", error=str(e)
            )

        end_time = time.perf_counter()
        
        ttft = (first_token_time - start_time) if first_token_time else 0
        total_time = end_time - start_time
        
        if first_token_time and last_token_time and last_token_time > first_token_time:
            generation_time = last_token_time - first_token_time
        else:
            generation_time = 0.000001
            
        throughput = token_count / generation_time if generation_time > 0 else 0
            
        return RequestMetrics(
            request_id=str(req_id),
            start_time=start_time,
            ttft=ttft,
            total_time=total_time,
            generation_time=generation_time,
            output_tokens=token_count,
            input_tokens=self.config.prompt_tokens,
            throughput=throughput
        )

    async def benchmark(self):
        prompt = self.generate_prompt(self.config.prompt_tokens)
        logger.info(f"Generated prompt with approx {self.config.prompt_tokens} tokens.")
        
        if self.config.warmup_runs > 0:
            logger.info(f"Starting warmup ({self.config.warmup_runs} runs)...")
            async with httpx.AsyncClient(headers=self.headers, timeout=self.timeout) as client:
                for i in range(self.config.warmup_runs):
                    await self.run_single_request(client, "-1", prompt)
            logger.info("Warmup complete.")

        results: List[RequestMetrics] = []
        
        start_global = time.perf_counter()

        if self.config.duration > 0:
            logger.info(f"Starting duration-based benchmark: {self.config.duration}s, mode={self.config.mode}, concurrency={self.config.concurrency}")
            end_time = time.time() + self.config.duration
            
            async def worker(worker_id):
                worker_results = []
                async with httpx.AsyncClient(headers=self.headers, timeout=self.timeout) as client:
                    while time.time() < end_time:
                         req_id = f"{worker_id}-{len(worker_results)}"
                         metrics = await self.run_single_request(client, req_id, prompt)
                         worker_results.append(metrics)
                return worker_results

            tasks = [worker(i) for i in range(self.config.concurrency)]
            nested_results = await asyncio.gather(*tasks)
            results = [item for sublist in nested_results for item in sublist]

        else:
            logger.info(f"Starting fixed-run benchmark: mode={self.config.mode}, runs={self.config.num_runs}, concurrency={self.config.concurrency}")
            semaphore = asyncio.Semaphore(self.config.concurrency)
            
            async def sem_task(i):
                async with semaphore:
                    async with httpx.AsyncClient(headers=self.headers, timeout=self.timeout) as client:
                        return await self.run_single_request(client, i, prompt)

            tasks = [sem_task(i) for i in range(self.config.num_runs)]
            results = await asyncio.gather(*tasks)
        
        total_duration = time.perf_counter() - start_global
        
        self.process_results(results, total_duration)

    def process_results(self, metrics: List[RequestMetrics], duration: float):
        successful = [m for m in metrics if m.status == "success"]
        failed = [m for m in metrics if m.status != "success"]
        
        if not successful:
            logger.error("All requests failed.")
            return

        if np:
            ttfts = np.array([m.ttft for m in successful]) * 1000
            throughputs = np.array([m.throughput for m in successful])
        else:
            ttfts = [m.ttft * 1000 for m in successful]
            throughputs = [m.throughput for m in successful]
        total_tokens = sum(m.output_tokens for m in successful)
        
        def safe_std(values):
            if np:
                return np.std(values) if len(values) > 1 else 0.0
            else:
                import math
                if len(values) <= 1:
                    return 0.0
                mean = sum(values) / len(values)
                var = sum((x - mean) ** 2 for x in values) / len(values)
                return math.sqrt(var)
        
        stats = {
            "mode": self.config.mode,
            "concurrency": self.config.concurrency,
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "total_duration_sec": duration,
            "total_tokens_generated": int(total_tokens),
            "ttft_ms": {
                "mean": (np.mean(ttfts) if np else sum(ttfts) / len(ttfts)) if len(ttfts) > 0 else 0,
                "p50": (np.percentile(ttfts, 50) if np else sorted(ttfts)[len(ttfts)//2]) if len(ttfts) > 0 else 0,
                "p95": (np.percentile(ttfts, 95) if np else sorted(ttfts)[int(len(ttfts)*0.95)-1]) if len(ttfts) > 0 else 0,
                "std": safe_std(ttfts)
            },
            "throughput_tokens_sec": {
                "mean": (np.mean(throughputs) if np else sum(throughputs) / len(throughputs)) if len(throughputs) > 0 else 0,
                "p50": (np.percentile(throughputs, 50) if np else sorted(throughputs)[len(throughputs)//2]) if len(throughputs) > 0 else 0,
                "p95": (np.percentile(throughputs, 95) if np else sorted(throughputs)[int(len(throughputs)*0.95)-1]) if len(throughputs) > 0 else 0,
                "std": safe_std(throughputs)
            },
            "overall_system_throughput": total_tokens / duration if duration else 0
        }
        
        self.print_report(stats)
        if self.config.output_file:
            self.save_json(stats, successful)

    def print_report(self, stats: Dict[str, Any]):
        print("\n" + "="*55)
        print(f"BENCHMARK REPORT: {self.config.model}")
        print(f"Mode: {self.config.mode} | Concurrency: {self.config.concurrency}")
        print("="*55)
        print(f"{'Metric':<30} | {'Value':<15}")
        print("-" * 48)
        
        ttft = stats["ttft_ms"]
        print(f"{'TTFT Mean (ms)':<30} | {ttft['mean']:.2f}")
        print(f"{'TTFT P50 (ms)':<30} | {ttft['p50']:.2f}")
        print(f"{'TTFT P95 (ms)':<30} | {ttft['p95']:.2f}")
        print("-" * 48)
        
        tps = stats["throughput_tokens_sec"]
        print(f"{'TPS Mean (tokens/s)':<30} | {tps['mean']:.2f}")
        print(f"{'TPS P95 (tokens/s)':<30} | {tps['p95']:.2f}")
        print("-" * 48)
        
        print(f"{'System Throughput (tokens/s)':<30} | {stats['overall_system_throughput']:.2f}")
        print(f"{'Total Requests':<30} | {stats['successful_requests']}")
        print(f"{'Failed Requests':<30} | {stats['failed_requests']}")
        print("="*55 + "\n")

    def save_json(self, stats: Dict[str, Any], runs: List[RequestMetrics]):
        run_data = [asdict(r) for r in runs]
        
        data = {
            "summary": stats,
            "config": {k:v for k,v in asdict(self.config).items() if k != "api_key"},
            "runs": run_data
        }
        
        def custom_serializer(obj):
            if np and isinstance(obj, np.generic):
                return obj.item()
            raise TypeError
            
        with open(self.config.output_file, 'w') as f:
            json.dump(data, f, indent=2, default=custom_serializer)
        logger.info(f"Results saved to {self.config.output_file}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("benchmark")

    parser = argparse.ArgumentParser(description="Production LLM Benchmark Tool")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API Base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API Key")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--mode", choices=["ttft", "throughput", "custom"], default="ttft", help="Benchmark mode")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("--runs", type=int, default=10, help="Total number of requests")
    parser.add_argument("--duration", type=int, default=0, help="Duration in seconds")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max tokens")
    parser.add_argument("--min-tokens", type=int, default=None, help="Minimum tokens to generate")
    parser.add_argument("--output", default="benchmark_results.json", help="Output JSON file path")
    
    args = parser.parse_args()
    
    if args.mode == "ttft":
        prompt_tokens = 3000
        max_tokens = 1
        concurrency = 1
        runs = max(args.runs, 10)
    elif args.mode == "throughput":
        prompt_tokens = 1000
        max_tokens = 1024
        concurrency = args.concurrency
        runs = max(args.runs, 5)
    else:
        prompt_tokens = 500
        max_tokens = 256
        concurrency = args.concurrency
        runs = args.runs

    if args.max_tokens:
        max_tokens = args.max_tokens
        
    config = BenchmarkConfig(
        api_base=args.base_url,
        api_key=args.api_key,
        model=args.model,
        mode=args.mode,
        num_runs=runs,
        warmup_runs=args.warmup,
        concurrency=concurrency,
        min_tokens=args.min_tokens,
        max_tokens=max_tokens,
        temperature=0.0,
        prompt_tokens=prompt_tokens,
        duration=args.duration,
        output_file=args.output
    )
    
    runner = BenchmarkRunner(config)
    asyncio.run(runner.benchmark())
