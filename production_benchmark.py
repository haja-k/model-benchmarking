"""
Production LLM Benchmark Tool
Industry-standard evaluation covering:
  - TTFT   : Time to First Token (latency)
  - TPOT   : Time Per Output Token (inter-token latency)
  - TTLT   : Time to Last Token (end-to-end latency)
  - ITL    : Inter-Token Latency (per-token generation speed)
  - TPS    : Throughput (tokens / second per request)
  - RPS    : Request Rate (requests / second system-wide)
  - System Throughput : total tokens / second across all concurrent workers
  - Concurrency Scaling : how throughput scales with parallelism
  - Error / Failure Rate
  - Percentile distribution : P50 / P90 / P95 / P99 for all timing metrics
"""

import argparse
import asyncio
import json
import logging
import math
import os
import re
import time
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple

import httpx
from dotenv import load_dotenv

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

load_dotenv()

DEFAULT_API_KEY = os.getenv("API_KEY")
DEFAULT_BASE_URL = os.getenv("API_BASE_URL")
DEFAULT_MODEL = os.getenv("MODEL")

# Allowed model identifiers for this deployment
KNOWN_MODELS = [
    "si-gpt-oss-120b",
    "si-qwen3-embedding-8b",
    "si-deepseek-3.2",
    "si-qwen3.5-27b",
    "si-qwen3-vl-30b",
    "si-qwen3.5-35b",
    "sains-llm-agentic",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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
    concurrency_sweep: bool = False
    sweep_levels: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    runs_per_sweep_level: int = 20


@dataclass
class RequestMetrics:
    request_id: str
    start_time: float
    ttft: float           # time-to-first-token  (seconds)
    tpot: float           # time-per-output-token (seconds) = generation_time / output_tokens
    ttlt: float           # time-to-last-token = total_time
    generation_time: float
    output_tokens: int
    input_tokens: int
    throughput: float     # tokens / sec for this request
    itl_values: List[float] = field(default_factory=list)  # per-token inter-token latencies (s)
    status: str = "success"
    error: Optional[str] = None
    concurrency_level: int = 1


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if HAS_NUMPY:
        return float(np.percentile(values, p))
    sv = sorted(values)
    idx = (len(sv) - 1) * p / 100.0
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sv):
        return sv[lo]
    return sv[lo] + (idx - lo) * (sv[hi] - sv[lo])


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    if HAS_NUMPY:
        return float(np.std(values))
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def compute_stats(values: List[float]) -> Dict[str, float]:
    return {
        "mean": _mean(values),
        "std":  _std(values),
        "min":  min(values) if values else 0.0,
        "p50":  _percentile(values, 50),
        "p90":  _percentile(values, 90),
        "p95":  _percentile(values, 95),
        "p99":  _percentile(values, 99),
        "max":  max(values) if values else 0.0,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        self.config.model = self.config.model.strip()
        self.timeout = httpx.Timeout(300.0, connect=30.0)

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def generate_prompt(self, target_tokens: int) -> str:
        """
        Generate a realistic mixed prompt that approximates `target_tokens`.
        Uses a paragraph of natural language to give the model realistic input
        (avoids pathological tokenization from simple repetition).
        """
        base = (
            "Artificial intelligence research has made remarkable strides over the past decade. "
            "Large language models demonstrate emergent capabilities in reasoning, code generation, "
            "summarisation, translation, and multi-step problem solving. "
            "Evaluating these systems rigorously requires measuring latency, throughput, "
            "token generation speed, error rates, and scalability simultaneously. "
        )
        chars_needed = target_tokens * 4
        repeats = (chars_needed // len(base)) + 1
        return (base * repeats)[:chars_needed]

    # ------------------------------------------------------------------
    # Single request
    # ------------------------------------------------------------------

    async def run_single_request(
        self,
        client: httpx.AsyncClient,
        req_id: str,
        prompt: str,
        concurrency_level: int = 1,
    ) -> RequestMetrics:
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens,
            "stream": True,
            "temperature": self.config.temperature,
            "stream_options": {"include_usage": True},
        }
        if self.config.min_tokens:
            payload["min_tokens"] = self.config.min_tokens

        base = self.config.api_base.rstrip("/")
        url = f"{base}/chat/completions"

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        prev_token_time: Optional[float] = None
        last_token_time: Optional[float] = None
        token_count = 0
        input_tokens = self.config.prompt_tokens
        itl_values: List[float] = []

        try:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    logger.error(
                        f"Request {req_id} failed: {response.status_code} – "
                        f"{error_body.decode('utf-8', errors='replace')}"
                    )
                    return RequestMetrics(
                        request_id=str(req_id),
                        start_time=start_time,
                        ttft=0, tpot=0, ttlt=0,
                        generation_time=0,
                        output_tokens=0, input_tokens=input_tokens,
                        throughput=0,
                        status="failed",
                        error=str(response.status_code),
                        concurrency_level=concurrency_level,
                    )

                async for line in response.aiter_lines():
                    if not line or not line.strip():
                        continue
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    now = time.perf_counter()

                    # First token
                    if first_token_time is None:
                        first_token_time = now
                        prev_token_time = now

                    # Usage chunk (vLLM includes at end with include_usage)
                    if "usage" in data and data["usage"]:
                        usage = data["usage"]
                        token_count_reported = usage.get("completion_tokens", 0)
                        if token_count_reported:
                            token_count = token_count_reported
                        input_tokens = usage.get("prompt_tokens", input_tokens)

                    # Content delta
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            last_token_time = now
                            token_count += 1
                            if prev_token_time is not None and now > prev_token_time:
                                itl_values.append(now - prev_token_time)
                            prev_token_time = now

        except Exception as exc:
            logger.error(f"Request {req_id} exception: {exc}")
            return RequestMetrics(
                request_id=str(req_id),
                start_time=start_time,
                ttft=0, tpot=0, ttlt=0,
                generation_time=0,
                output_tokens=0, input_tokens=input_tokens,
                throughput=0,
                status="error",
                error=str(exc),
                concurrency_level=concurrency_level,
            )

        end_time = time.perf_counter()

        ttft = (first_token_time - start_time) if first_token_time else 0.0
        ttlt = end_time - start_time

        if first_token_time and last_token_time and last_token_time > first_token_time:
            generation_time = last_token_time - first_token_time
        else:
            generation_time = max(ttlt - ttft, 1e-6)

        # TPOT: average time between consecutive tokens
        tpot = _mean(itl_values) if itl_values else (generation_time / max(token_count, 1))
        throughput = token_count / generation_time if generation_time > 0 else 0.0

        return RequestMetrics(
            request_id=str(req_id),
            start_time=start_time,
            ttft=ttft,
            tpot=tpot,
            ttlt=ttlt,
            generation_time=generation_time,
            output_tokens=token_count,
            input_tokens=input_tokens,
            throughput=throughput,
            itl_values=itl_values,
            concurrency_level=concurrency_level,
        )

    # ------------------------------------------------------------------
    # Benchmark orchestration
    # ------------------------------------------------------------------

    async def _run_fixed(
        self,
        prompt: str,
        num_runs: int,
        concurrency: int,
        concurrency_level: int = 1,
    ) -> Tuple[List[RequestMetrics], float]:
        semaphore = asyncio.Semaphore(concurrency)

        async def sem_task(i: int) -> RequestMetrics:
            async with semaphore:
                async with httpx.AsyncClient(headers=self.headers, timeout=self.timeout) as client:
                    return await self.run_single_request(client, str(i), prompt, concurrency_level)

        t0 = time.perf_counter()
        results = list(await asyncio.gather(*[sem_task(i) for i in range(num_runs)]))
        duration = time.perf_counter() - t0
        return results, duration

    async def _run_duration(
        self,
        prompt: str,
        duration_sec: int,
        concurrency: int,
    ) -> Tuple[List[RequestMetrics], float]:
        end_ts = time.time() + duration_sec

        async def worker(wid: int) -> List[RequestMetrics]:
            worker_results: List[RequestMetrics] = []
            async with httpx.AsyncClient(headers=self.headers, timeout=self.timeout) as client:
                while time.time() < end_ts:
                    rid = f"{wid}-{len(worker_results)}"
                    m = await self.run_single_request(client, rid, prompt, concurrency)
                    worker_results.append(m)
            return worker_results

        t0 = time.perf_counter()
        nested = await asyncio.gather(*[worker(i) for i in range(concurrency)])
        duration = time.perf_counter() - t0
        results = [item for sub in nested for item in sub]
        return results, duration

    async def benchmark(self) -> Dict[str, Any]:
        # Announce the HTML output path upfront so the user knows where to look
        html_path = self._build_html_path()
        logger.info(f"HTML report will be saved → {html_path}")

        prompt = self.generate_prompt(self.config.prompt_tokens)
        logger.info(f"Prompt generated: ~{self.config.prompt_tokens} input tokens")

        # Warmup
        if self.config.warmup_runs > 0:
            logger.info(f"Warming up ({self.config.warmup_runs} requests)…")
            async with httpx.AsyncClient(headers=self.headers, timeout=self.timeout) as client:
                for i in range(self.config.warmup_runs):
                    await self.run_single_request(client, f"warmup-{i}", prompt)
            logger.info("Warmup complete.")

        all_results: List[RequestMetrics] = []
        phase_stats: List[Dict[str, Any]] = []

        # ---- Main benchmark phase ----
        if self.config.duration > 0:
            logger.info(
                f"Duration benchmark: {self.config.duration}s | "
                f"concurrency={self.config.concurrency}"
            )
            results, dur = await self._run_duration(
                prompt, self.config.duration, self.config.concurrency
            )
        else:
            logger.info(
                f"Fixed benchmark: {self.config.num_runs} runs | "
                f"concurrency={self.config.concurrency}"
            )
            results, dur = await self._run_fixed(
                prompt, self.config.num_runs, self.config.concurrency, self.config.concurrency
            )

        all_results.extend(results)
        phase_stats.append(self._compute_phase_stats(results, dur, self.config.concurrency))
        self.print_report(phase_stats[0])

        # ---- Concurrency sweep ----
        sweep_stats: List[Dict[str, Any]] = []
        if self.config.concurrency_sweep:
            logger.info("Starting concurrency sweep…")
            for level in self.config.sweep_levels:
                logger.info(f"  Sweep concurrency={level}")
                s_results, s_dur = await self._run_fixed(
                    prompt, self.config.runs_per_sweep_level, level, level
                )
                s = self._compute_phase_stats(s_results, s_dur, level)
                sweep_stats.append(s)
                logger.info(
                    f"  c={level}: system_tps={s['overall_system_throughput']:.1f} | "
                    f"ttft_p95={s['ttft_ms']['p95']:.1f}ms"
                )
            all_results.extend(s_results)  # type: ignore

        # ---- HTML report ----
        self._save_html(phase_stats[0], sweep_stats, all_results, html_path)

        # ---- JSON output ----
        if self.config.output_file:
            self._save_json(phase_stats[0], all_results)

        return phase_stats[0]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _compute_phase_stats(
        self,
        metrics: List[RequestMetrics],
        duration: float,
        concurrency: int,
    ) -> Dict[str, Any]:
        successful = [m for m in metrics if m.status == "success"]
        failed = [m for m in metrics if m.status != "success"]

        if not successful:
            logger.error("No successful requests in this phase.")
            return {}

        ttfts_ms   = [m.ttft   * 1000 for m in successful]
        tpots_ms   = [m.tpot   * 1000 for m in successful]
        ttlts_ms   = [m.ttlt   * 1000 for m in successful]
        throughputs = [m.throughput for m in successful]
        output_tokens_list = [m.output_tokens for m in successful]

        # Flatten all ITL values for a global ITL distribution
        all_itl_ms = [v * 1000 for m in successful for v in m.itl_values]

        total_tokens = sum(output_tokens_list)
        rps = len(successful) / duration if duration else 0.0

        return {
            "concurrency": concurrency,
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "error_rate_pct": 100.0 * len(failed) / len(metrics) if metrics else 0.0,
            "total_duration_sec": round(duration, 3),
            "total_tokens_generated": int(total_tokens),
            "rps": round(rps, 3),
            "overall_system_throughput": round(total_tokens / duration, 2) if duration else 0.0,
            "ttft_ms":       compute_stats(ttfts_ms),
            "tpot_ms":       compute_stats(tpots_ms),
            "ttlt_ms":       compute_stats(ttlts_ms),
            "itl_ms":        compute_stats(all_itl_ms),
            "throughput_tps": compute_stats(throughputs),
            "output_tokens": compute_stats(output_tokens_list),
        }

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------

    def print_report(self, stats: Dict[str, Any]):
        if not stats:
            return
        W = 58
        print("\n" + "=" * W)
        print(f"  BENCHMARK REPORT  —  {self.config.model}")
        print(f"  Mode: {self.config.mode}  |  Concurrency: {stats['concurrency']}")
        print("=" * W)

        def row(label: str, value: str):
            print(f"  {label:<32} {value}")

        def stat_rows(label: str, s: Dict[str, float], unit: str):
            row(f"{label} mean",  f"{s['mean']:.2f} {unit}")
            row(f"{label} P50",   f"{s['p50']:.2f} {unit}")
            row(f"{label} P90",   f"{s['p90']:.2f} {unit}")
            row(f"{label} P95",   f"{s['p95']:.2f} {unit}")
            row(f"{label} P99",   f"{s['p99']:.2f} {unit}")
            row(f"{label} std",   f"{s['std']:.2f} {unit}")
            print("  " + "-" * (W - 2))

        print("  " + "-" * (W - 2))
        stat_rows("TTFT", stats["ttft_ms"], "ms")
        stat_rows("TPOT", stats["tpot_ms"], "ms")
        stat_rows("TTLT (E2E latency)", stats["ttlt_ms"], "ms")
        stat_rows("ITL (inter-token)", stats["itl_ms"] if stats["itl_ms"]["mean"] else stats["tpot_ms"], "ms")
        stat_rows("Per-req TPS", stats["throughput_tps"], "tok/s")
        stat_rows("Output tokens", stats["output_tokens"], "tok")

        print(f"  {'System throughput':<32} {stats['overall_system_throughput']:.2f} tok/s")
        print(f"  {'Request rate (RPS)':<32} {stats['rps']:.3f} req/s")
        print(f"  {'Successful requests':<32} {stats['successful_requests']}")
        print(f"  {'Failed requests':<32} {stats['failed_requests']}")
        print(f"  {'Error rate':<32} {stats['error_rate_pct']:.1f}%")
        print(f"  {'Total duration':<32} {stats['total_duration_sec']:.2f}s")
        print(f"  {'Total tokens generated':<32} {stats['total_tokens_generated']}")
        print("=" * W + "\n")

    # ------------------------------------------------------------------
    # HTML output path
    # ------------------------------------------------------------------

    def _build_html_path(self) -> str:
      ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
      # Sanitize the model name for use in a filename
      safe_model = re.sub(r"[^\w\-.]", "_", self.config.model)
      results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
      os.makedirs(results_dir, exist_ok=True)
      return os.path.join(results_dir, f"{ts}_{safe_model}_benchmark.html")

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def _save_html(
        self,
        stats: Dict[str, Any],
        sweep_stats: List[Dict[str, Any]],
        all_results: List[RequestMetrics],
        path: str,
    ):
        # If all requests failed, write a minimal error report instead of silently skipping
        if not stats:
            failed = [m for m in all_results if m.status != "success"]
            first_err = failed[0].error if failed else "unknown"
            error_html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"/>
<title>Benchmark Failed — {self.config.model}</title>
<style>body{{font-family:system-ui;background:#0f172a;color:#e2e8f0;padding:2rem;}}
h1{{color:#f87171;}}pre{{background:#1e293b;padding:1rem;border-radius:8px;color:#fca5a5;}}</style>
</head><body>
<h1>Benchmark Failed</h1>
<p>Model: <strong>{self.config.model}</strong> — all requests returned errors.</p>
<pre>{first_err}</pre>
<p>Allowed models for this deployment:</p>
<ul>{''.join(f'<li>{m}</li>' for m in KNOWN_MODELS)}</ul>
</body></html>"""
            with open(path, "w", encoding="utf-8") as f:
                f.write(error_html)
            logger.error(f"All requests failed. Error report saved → {path}")
            return

        run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        successful = [m for m in all_results if m.status == "success"]

        # ---- Per-request time-series data (main phase only) ----
        main_results = [m for m in successful if m.concurrency_level == self.config.concurrency]

        req_ids     = [m.request_id for m in main_results]
        ttft_list   = [round(m.ttft  * 1000, 2) for m in main_results]
        tpot_list   = [round(m.tpot  * 1000, 2) for m in main_results]
        ttlt_list   = [round(m.ttlt  * 1000, 2) for m in main_results]
        tps_list    = [round(m.throughput, 2)    for m in main_results]

        # ---- Sweep data ----
        sweep_labels  = [str(s["concurrency"]) for s in sweep_stats]
        sweep_sys_tps = [s["overall_system_throughput"] for s in sweep_stats]
        sweep_rps     = [s["rps"] for s in sweep_stats]
        sweep_ttft_p95 = [s["ttft_ms"]["p95"] for s in sweep_stats]

        # Percentile bar chart data for main phase
        pct_labels  = ["P50", "P90", "P95", "P99"]
        ttft_pcts   = [round(stats["ttft_ms"][k], 2)    for k in ["p50","p90","p95","p99"]]
        tpot_pcts   = [round(stats["tpot_ms"][k], 2)    for k in ["p50","p90","p95","p99"]]
        ttlt_pcts   = [round(stats["ttlt_ms"][k], 2)    for k in ["p50","p90","p95","p99"]]

        # Config table rows
        cfg = self.config
        config_rows = [
            ("Model",              cfg.model),
            ("Benchmark Mode",     cfg.mode),
            ("Concurrency",        str(cfg.concurrency)),
            ("Total Runs",         str(cfg.num_runs)),
            ("Warmup Runs",        str(cfg.warmup_runs)),
            ("Max Output Tokens",  str(cfg.max_tokens)),
            ("Min Output Tokens",  str(cfg.min_tokens) if cfg.min_tokens else "—"),
            ("Prompt Tokens",      str(cfg.prompt_tokens)),
            ("Temperature",        str(cfg.temperature)),
            ("Duration (s)",       str(cfg.duration) if cfg.duration else "—"),
            ("API Base URL",       cfg.api_base),
            ("Report Generated",   run_ts),
        ]

        def fmt_cfg_rows(rows):
            return "\n".join(
                f'<tr><td class="cfg-key">{k}</td><td class="cfg-val">{v}</td></tr>'
                for k, v in rows
            )

        sweep_section = ""
        if sweep_stats:
            sweep_section = f"""
        <div class="card">
          <h2>Concurrency Scaling</h2>
          <div class="chart-wrap"><canvas id="chartSweepTps"></canvas></div>
          <div class="chart-wrap"><canvas id="chartSweepRps"></canvas></div>
          <div class="chart-wrap"><canvas id="chartSweepTtft"></canvas></div>
        </div>
"""
            sweep_js = f"""
  // ---- Concurrency sweep charts ----
  new Chart(document.getElementById('chartSweepTps'), {{
    type: 'line',
    data: {{
      labels: {json.dumps(sweep_labels)},
      datasets: [{{
        label: 'System Throughput (tok/s)',
        data: {json.dumps(sweep_sys_tps)},
        borderColor: '#6366f1', backgroundColor: 'rgba(99,102,241,0.15)',
        fill: true, tension: 0.3, pointRadius: 5
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{ title: {{ display: true, text: 'System Throughput vs Concurrency' }} }},
      scales: {{
        x: {{ title: {{ display: true, text: 'Concurrency Level' }} }},
        y: {{ title: {{ display: true, text: 'tokens / second' }}, beginAtZero: true }}
      }}
    }}
  }});
  new Chart(document.getElementById('chartSweepRps'), {{
    type: 'line',
    data: {{
      labels: {json.dumps(sweep_labels)},
      datasets: [{{
        label: 'Requests / second (RPS)',
        data: {json.dumps(sweep_rps)},
        borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.15)',
        fill: true, tension: 0.3, pointRadius: 5
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{ title: {{ display: true, text: 'RPS vs Concurrency' }} }},
      scales: {{
        x: {{ title: {{ display: true, text: 'Concurrency Level' }} }},
        y: {{ title: {{ display: true, text: 'requests / second' }}, beginAtZero: true }}
      }}
    }}
  }});
  new Chart(document.getElementById('chartSweepTtft'), {{
    type: 'line',
    data: {{
      labels: {json.dumps(sweep_labels)},
      datasets: [{{
        label: 'TTFT P95 (ms)',
        data: {json.dumps(sweep_ttft_p95)},
        borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.15)',
        fill: true, tension: 0.3, pointRadius: 5
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{ title: {{ display: true, text: 'TTFT P95 vs Concurrency' }} }},
      scales: {{
        x: {{ title: {{ display: true, text: 'Concurrency Level' }} }},
        y: {{ title: {{ display: true, text: 'ms' }}, beginAtZero: true }}
      }}
    }}
  }});
"""
        else:
            sweep_js = ""

        # Build KPI cards
        def kpi(label, value, unit=""):
            return f"""<div class="kpi"><div class="kpi-label">{label}</div>
<div class="kpi-value">{value}<span class="kpi-unit"> {unit}</span></div></div>"""

        kpis = "".join([
            kpi("TTFT P50",           f"{stats['ttft_ms']['p50']:.1f}", "ms"),
            kpi("TTFT P95",           f"{stats['ttft_ms']['p95']:.1f}", "ms"),
            kpi("TTFT P99",           f"{stats['ttft_ms']['p99']:.1f}", "ms"),
            kpi("TPOT Mean",          f"{stats['tpot_ms']['mean']:.1f}", "ms"),
            kpi("TTLT P95",           f"{stats['ttlt_ms']['p95']:.1f}", "ms"),
            kpi("System Throughput",  f"{stats['overall_system_throughput']:.1f}", "tok/s"),
            kpi("Per-req TPS P50",    f"{stats['throughput_tps']['p50']:.1f}", "tok/s"),
            kpi("RPS",                f"{stats['rps']:.3f}", "req/s"),
            kpi("Error Rate",         f"{stats['error_rate_pct']:.1f}", "%"),
            kpi("Total Requests",     str(stats["successful_requests"]), "ok"),
        ])

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>LLM Benchmark — {cfg.model}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #0f172a; color: #e2e8f0;
      padding: 2rem;
    }}
    h1 {{ font-size: 1.7rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.25rem; }}
    h2 {{ font-size: 1.1rem; font-weight: 600; color: #94a3b8; margin-bottom: 1rem; }}
    .subtitle {{ color: #64748b; font-size: 0.85rem; margin-bottom: 2rem; }}
    .card {{
      background: #1e293b; border-radius: 12px;
      padding: 1.5rem; margin-bottom: 1.5rem;
      box-shadow: 0 2px 12px rgba(0,0,0,0.4);
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 1rem; margin-bottom: 1.5rem;
    }}
    .kpi {{
      background: #0f172a; border-radius: 10px;
      padding: 1rem 1.2rem;
      border: 1px solid #334155;
    }}
    .kpi-label {{ font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
    .kpi-value {{ font-size: 1.6rem; font-weight: 700; color: #f1f5f9; margin-top: 0.25rem; }}
    .kpi-unit  {{ font-size: 0.85rem; font-weight: 400; color: #94a3b8; }}
    .chart-wrap {{ position: relative; height: 280px; margin-bottom: 1.5rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    th, td {{ padding: 0.55rem 0.75rem; border-bottom: 1px solid #334155; text-align: left; }}
    th {{ color: #94a3b8; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; }}
    tr:last-child td {{ border-bottom: none; }}
    .cfg-key {{ color: #94a3b8; width: 40%; }}
    .cfg-val {{ color: #f1f5f9; }}
    .badge-ok  {{ background: #14532d; color: #86efac; padding: 0.2rem 0.5rem; border-radius: 999px; font-size: 0.75rem; }}
    .badge-err {{ background: #7f1d1d; color: #fca5a5; padding: 0.2rem 0.5rem; border-radius: 999px; font-size: 0.75rem; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
    @media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>LLM Benchmark Report</h1>
  <p class="subtitle">Model: <strong>{cfg.model}</strong> &nbsp;|&nbsp; {run_ts} &nbsp;|&nbsp; Mode: {cfg.mode}</p>

  <!-- KPI summary -->
  <div class="kpi-grid">{kpis}</div>

  <!-- Testing parameters -->
  <div class="card">
    <h2>Testing Parameters</h2>
    <table>
      <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
      <tbody>{fmt_cfg_rows(config_rows)}</tbody>
    </table>
  </div>

  <!-- Latency percentile charts -->
  <div class="card">
    <h2>Latency Percentile Distribution</h2>
    <div class="two-col">
      <div class="chart-wrap"><canvas id="chartTtftPct"></canvas></div>
      <div class="chart-wrap"><canvas id="chartTpotPct"></canvas></div>
    </div>
    <div class="chart-wrap" style="max-width:600px;margin:0 auto;">
      <canvas id="chartTtltPct"></canvas>
    </div>
  </div>

  <!-- Time-series scatter -->
  <div class="card">
    <h2>Per-Request Latency Over Time</h2>
    <div class="chart-wrap"><canvas id="chartTimeSeries"></canvas></div>
  </div>

  <!-- Throughput histogram -->
  <div class="card">
    <h2>Per-Request Throughput Distribution</h2>
    <div class="chart-wrap"><canvas id="chartTpsHist"></canvas></div>
  </div>

  <!-- Detailed stats table -->
  <div class="card">
    <h2>Detailed Statistics</h2>
    <table>
      <thead>
        <tr>
          <th>Metric</th><th>Mean</th><th>Std</th>
          <th>P50</th><th>P90</th><th>P95</th><th>P99</th><th>Max</th>
        </tr>
      </thead>
      <tbody>
        {_html_stat_row("TTFT (ms)",            stats["ttft_ms"])}
        {_html_stat_row("TPOT (ms)",            stats["tpot_ms"])}
        {_html_stat_row("TTLT / E2E (ms)",      stats["ttlt_ms"])}
        {_html_stat_row("ITL (ms)",             stats["itl_ms"])}
        {_html_stat_row("Per-req TPS (tok/s)",  stats["throughput_tps"])}
        {_html_stat_row("Output tokens",        stats["output_tokens"])}
      </tbody>
    </table>
    <br/>
    <table>
      <tbody>
        <tr><td class="cfg-key">System Throughput</td><td class="cfg-val">{stats['overall_system_throughput']:.2f} tok/s</td></tr>
        <tr><td class="cfg-key">RPS</td><td class="cfg-val">{stats['rps']:.4f} req/s</td></tr>
        <tr><td class="cfg-key">Total Requests</td><td class="cfg-val">{stats['successful_requests']} <span class="badge-ok">ok</span> &nbsp; {stats['failed_requests']} <span class="badge-err">err</span></td></tr>
        <tr><td class="cfg-key">Error Rate</td><td class="cfg-val">{stats['error_rate_pct']:.2f}%</td></tr>
      </tbody>
    </table>
  </div>

  {sweep_section}

  <script>
  // ---- Latency percentile bar charts ----
  const pctLabels = {json.dumps(pct_labels)};

  new Chart(document.getElementById('chartTtftPct'), {{
    type: 'bar',
    data: {{
      labels: pctLabels,
      datasets: [{{
        label: 'TTFT (ms)',
        data: {json.dumps(ttft_pcts)},
        backgroundColor: ['#6366f1','#818cf8','#a5b4fc','#c7d2fe'],
        borderRadius: 6
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ title: {{ display: true, text: 'TTFT Percentiles' }}, legend: {{ display: false }} }},
      scales: {{ y: {{ title: {{ display: true, text: 'ms' }}, beginAtZero: true }} }}
    }}
  }});

  new Chart(document.getElementById('chartTpotPct'), {{
    type: 'bar',
    data: {{
      labels: pctLabels,
      datasets: [{{
        label: 'TPOT (ms)',
        data: {json.dumps(tpot_pcts)},
        backgroundColor: ['#10b981','#34d399','#6ee7b7','#a7f3d0'],
        borderRadius: 6
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ title: {{ display: true, text: 'TPOT Percentiles' }}, legend: {{ display: false }} }},
      scales: {{ y: {{ title: {{ display: true, text: 'ms' }}, beginAtZero: true }} }}
    }}
  }});

  new Chart(document.getElementById('chartTtltPct'), {{
    type: 'bar',
    data: {{
      labels: pctLabels,
      datasets: [{{
        label: 'TTLT (ms)',
        data: {json.dumps(ttlt_pcts)},
        backgroundColor: ['#f59e0b','#fbbf24','#fcd34d','#fde68a'],
        borderRadius: 6
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ title: {{ display: true, text: 'TTLT / E2E Latency Percentiles' }}, legend: {{ display: false }} }},
      scales: {{ y: {{ title: {{ display: true, text: 'ms' }}, beginAtZero: true }} }}
    }}
  }});

  // ---- Time-series (per request) ----
  const tsLabels = {json.dumps(req_ids)};
  new Chart(document.getElementById('chartTimeSeries'), {{
    type: 'line',
    data: {{
      labels: tsLabels,
      datasets: [
        {{
          label: 'TTFT (ms)',
          data: {json.dumps(ttft_list)},
          borderColor: '#6366f1', backgroundColor: 'transparent',
          pointRadius: 3, tension: 0.2
        }},
        {{
          label: 'TTLT (ms)',
          data: {json.dumps(ttlt_list)},
          borderColor: '#f59e0b', backgroundColor: 'transparent',
          pointRadius: 3, tension: 0.2
        }},
        {{
          label: 'TPOT (ms)',
          data: {json.dumps(tpot_list)},
          borderColor: '#10b981', backgroundColor: 'transparent',
          pointRadius: 3, tension: 0.2
        }}
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ title: {{ display: true, text: 'Latency per Request' }} }},
      scales: {{
        x: {{ title: {{ display: true, text: 'Request ID' }} }},
        y: {{ title: {{ display: true, text: 'ms' }}, beginAtZero: true }}
      }}
    }}
  }});

  // ---- TPS histogram ----
  const tpsRaw = {json.dumps(tps_list)};
  const tpsBuckets = 20;
  const tpsMin = Math.min(...tpsRaw);
  const tpsMax = Math.max(...tpsRaw);
  const tpsStep = (tpsMax - tpsMin) / tpsBuckets || 1;
  const tpsEdges = Array.from({{length: tpsBuckets}}, (_, i) => tpsMin + i * tpsStep);
  const tpsCounts = new Array(tpsBuckets).fill(0);
  tpsRaw.forEach(v => {{
    const idx = Math.min(Math.floor((v - tpsMin) / tpsStep), tpsBuckets - 1);
    tpsCounts[idx]++;
  }});
  new Chart(document.getElementById('chartTpsHist'), {{
    type: 'bar',
    data: {{
      labels: tpsEdges.map(v => v.toFixed(1) + '+'),
      datasets: [{{
        label: 'Requests',
        data: tpsCounts,
        backgroundColor: 'rgba(99,102,241,0.7)',
        borderColor: '#6366f1',
        borderWidth: 1,
        borderRadius: 4
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ title: {{ display: true, text: 'Per-Request Throughput Histogram (tok/s)' }}, legend: {{ display: false }} }},
      scales: {{
        x: {{ title: {{ display: true, text: 'Throughput (tok/s)' }} }},
        y: {{ title: {{ display: true, text: 'Count' }}, beginAtZero: true }}
      }}
    }}
  }});

  {sweep_js}
  </script>
</body>
</html>
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"HTML report saved → {path}")

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------

    def _save_json(self, stats: Dict[str, Any], runs: List[RequestMetrics]):
        run_data = []
        for r in runs:
            d = asdict(r)
            d.pop("itl_values", None)  # omit verbose per-token data from JSON
            run_data.append(d)

        data = {
            "summary": stats,
            "config": {k: v for k, v in asdict(self.config).items() if k != "api_key"},
            "runs": run_data,
        }

        def _default(obj):
            if HAS_NUMPY and isinstance(obj, np.generic):
                return obj.item()
            raise TypeError

        with open(self.config.output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_default)
        logger.info(f"JSON results saved → {self.config.output_file}")


# ---------------------------------------------------------------------------
# HTML helper
# ---------------------------------------------------------------------------

def _html_stat_row(label: str, s: Dict[str, float]) -> str:
    def fmt(v: float) -> str:
        return f"{v:.2f}"
    return (
        f'<tr><td>{label}</td>'
        f'<td>{fmt(s["mean"])}</td><td>{fmt(s["std"])}</td>'
        f'<td>{fmt(s["p50"])}</td><td>{fmt(s["p90"])}</td>'
        f'<td>{fmt(s["p95"])}</td><td>{fmt(s["p99"])}</td>'
        f'<td>{fmt(s["max"])}</td></tr>'
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("benchmark")

    parser = argparse.ArgumentParser(
        description="Production LLM Benchmark Tool — industry-standard evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url",    default=DEFAULT_BASE_URL, help="API Base URL (OpenAI-compatible)")
    parser.add_argument("--api-key",     default=DEFAULT_API_KEY,  help="API key")
    parser.add_argument("--model",       default=DEFAULT_MODEL,    help="Model identifier")
    parser.add_argument("--mode",
        choices=["ttft", "throughput", "balanced", "custom"],
        default="balanced",
        help=(
            "ttft       – optimise for measuring first-token latency (short output)\n"
            "throughput – optimise for measuring sustained token generation\n"
            "balanced   – balanced latency + throughput (recommended)\n"
            "custom     – use --prompt-tokens / --max-tokens / --concurrency directly"
        ),
    )
    parser.add_argument("--concurrency",   type=int, default=1,    help="Concurrent workers")
    parser.add_argument("--runs",          type=int, default=20,   help="Total requests to send")
    parser.add_argument("--duration",      type=int, default=0,    help="Run for N seconds instead of fixed runs")
    parser.add_argument("--warmup",        type=int, default=2,    help="Warmup requests (excluded from stats)")
    parser.add_argument("--max-tokens",    type=int, default=None, help="Override max output tokens")
    parser.add_argument("--min-tokens",    type=int, default=None, help="Force minimum output tokens")
    parser.add_argument("--prompt-tokens", type=int, default=None, help="Override input prompt length (tokens)")
    parser.add_argument("--output",        default=None,           help="Save raw results to JSON file")
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run concurrency sweep to measure scaling (1, 2, 4, 8, 16 workers)",
    )
    parser.add_argument(
        "--sweep-levels", default="1,2,4,8,16",
        help="Comma-separated concurrency levels for sweep (default: 1,2,4,8,16)",
    )
    parser.add_argument(
        "--sweep-runs", type=int, default=20,
        help="Requests per concurrency level during sweep",
    )

    args = parser.parse_args()

    # ---- Validate model name against allowed list ----
    if args.model not in KNOWN_MODELS:
        print(
            f"\nERROR: Model '{args.model}' is not in the allowed list.\n"
            f"Allowed models:\n" + "\n".join(f"  {m}" for m in KNOWN_MODELS) + "\n"
            f"\nSet --model to one of the above, or update MODEL= in your .env file.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Mode presets (industry-aligned) ----
    if args.mode == "ttft":
        # Measure time-to-first-token: large input, minimal output, serial
        prompt_tokens = args.prompt_tokens or 3000
        max_tokens    = 1
        concurrency   = 1
        runs          = max(args.runs, 20)
    elif args.mode == "throughput":
        # Sustained generation: moderate input, large output, high concurrency
        prompt_tokens = args.prompt_tokens or 500
        max_tokens    = args.max_tokens or 1024
        concurrency   = args.concurrency
        runs          = max(args.runs, 20)
    elif args.mode == "balanced":
        # Real-world mix: moderate I/O, moderate concurrency
        prompt_tokens = args.prompt_tokens or 1000
        max_tokens    = args.max_tokens or 512
        concurrency   = args.concurrency
        runs          = max(args.runs, 20)
    else:  # custom
        prompt_tokens = args.prompt_tokens or 500
        max_tokens    = args.max_tokens or 256
        concurrency   = args.concurrency
        runs          = args.runs

    if args.max_tokens:
        max_tokens = args.max_tokens

    sweep_levels = [int(x) for x in args.sweep_levels.split(",") if x.strip().isdigit()]

    config = BenchmarkConfig(
        api_base          = args.base_url,
        api_key           = args.api_key,
        model             = args.model,
        mode              = args.mode,
        num_runs          = runs,
        warmup_runs       = args.warmup,
        concurrency       = concurrency,
        min_tokens        = args.min_tokens,
        max_tokens        = max_tokens,
        temperature       = 0.0,
        prompt_tokens     = prompt_tokens,
        duration          = args.duration,
        output_file       = args.output,
        concurrency_sweep = args.sweep,
        sweep_levels      = sweep_levels,
        runs_per_sweep_level = args.sweep_runs,
    )

    runner = BenchmarkRunner(config)
    asyncio.run(runner.benchmark())
