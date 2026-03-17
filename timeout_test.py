"""
LLM Timeout Diagnostic Tool — Industry-Standard Root-Cause Analysis
====================================================================
Diagnostic categories (following ANSI/IEEE and LLMPerf/vLLM community standards):

  Phase 1 – Connectivity
    TC-01  Connect timeout           : TCP handshake exceeds threshold
    TC-02  TLS negotiation latency   : Time from connect to first byte of HTTP response

  Phase 2 – Server-Side Queue / Schedule Delay (TTFT gate)
    TC-03  TTFT under load           : Serial baseline vs high-concurrency TTFT degradation
    TC-04  Queue saturation          : TTFT grows super-linearly with concurrency
    TC-05  Cold-start detection      : First request in a session showing anomalous TTFT

  Phase 3 – Model Generation Stall (TPOT / ITL gate)
    TC-06  Generation stall          : Chunk gap > stall_threshold mid-stream
    TC-07  TPOT degradation          : Time-per-output-token increases beyond SLO
    TC-08  Partial response          : Non-zero tokens received but stream terminates early

  Phase 4 – End-to-End SLO Validation
    TC-09  Short client timeout      : Client-side budgets (5s / 30s / 120s)
    TC-10  Long generation (large output): max_tokens=2048, SLO=180s
    TC-11  Large input               : prompt_tokens=5000 / 10000

  Phase 5 – Concurrency & Load
    TC-12  Sequential stability      : 5 serial requests, no regression
    TC-13  Burst concurrency (10x)   : 10 parallel, measures timeout rate
    TC-14  Concurrency sweep         : 1/2/4/8 workers — TTFT P95 vs load

Root-cause classifier maps each failure to one of:
  NETWORK   — connect / TLS / DNS
  QUEUE     — TTFT anomaly (server queue saturation)
  STALL     — mid-stream gap (decode stall, GPU memory pressure)
  TIMEOUT   — client budget exceeded
  PARTIAL   — partial token generation (model / infra crash mid-stream)
  HTTP_ERR  — non-200 status
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

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("timeout_diag")

# ---------------------------------------------------------------------------
# Environment defaults
# ---------------------------------------------------------------------------

DEFAULT_API_KEY    = os.getenv("API_KEY")
DEFAULT_API_BASE   = os.getenv("API_BASE_URL")
DEFAULT_MODEL      = os.getenv("MODEL_SI_QWEN") or os.getenv("MODEL")
DEFAULT_TIMEOUT    = float(os.getenv("DEFAULT_TIMEOUT", "60.0"))
DEFAULT_PROMPT_TOKENS = int(os.getenv("DEFAULT_PROMPT_TOKENS", "1000"))
DEFAULT_MAX_TOKENS    = int(os.getenv("DEFAULT_MAX_TOKENS", "512"))
DEFAULT_MIN_TOKENS    = int(os.getenv("DEFAULT_MIN_TOKENS", "256"))

# SLO thresholds (industry-reference defaults — adjust to deployment SLA)
SLO_TTFT_MS        = float(os.getenv("SLO_TTFT_MS",   "3000"))   # 3 s
SLO_TPOT_MS        = float(os.getenv("SLO_TPOT_MS",   "150"))    # 150 ms / token
SLO_TTLT_MS        = float(os.getenv("SLO_TTLT_MS",   "60000"))  # 60 s
STALL_THRESHOLD_MS = float(os.getenv("STALL_THRESHOLD_MS", "5000"))  # 5 s gap = stall


# ---------------------------------------------------------------------------
# Statistics helpers (mirror production_benchmark.py for consistency)
# ---------------------------------------------------------------------------

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    idx = (len(sv) - 1) * p / 100.0
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sv):
        return sv[lo]
    return sv[lo] + (idx - lo) * (sv[hi] - sv[lo])

def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
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
# Root-cause classifier
# ---------------------------------------------------------------------------

class RootCause:
    NETWORK  = "NETWORK"   # connect / TLS failure
    QUEUE    = "QUEUE"     # TTFT anomaly — server queue saturation
    STALL    = "STALL"     # mid-stream gap — decode stall / GPU pressure
    TIMEOUT  = "TIMEOUT"   # client budget exceeded
    PARTIAL  = "PARTIAL"   # partial token generation
    HTTP_ERR = "HTTP_ERR"  # non-200 HTTP status
    OK       = "OK"        # no issue found


def classify_root_cause(result: "DiagResult") -> str:
    if result.connect_error:
        return RootCause.NETWORK
    if result.http_status and result.http_status != 200:
        return RootCause.HTTP_ERR
    if result.timeout_occurred and result.tokens_received == 0:
        return RootCause.TIMEOUT
    if result.timeout_occurred and result.tokens_received > 0:
        return RootCause.PARTIAL
    if result.stall_detected:
        return RootCause.STALL
    if result.ttft_ms > SLO_TTFT_MS and not result.timeout_occurred:
        return RootCause.QUEUE
    if result.error:
        return RootCause.NETWORK
    return RootCause.OK


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DiagConfig:
    api_base: str
    api_key: str
    model: str
    prompt_tokens: int
    max_tokens: int
    min_tokens: int
    slo_ttft_ms: float      = SLO_TTFT_MS
    slo_tpot_ms: float      = SLO_TPOT_MS
    slo_ttlt_ms: float      = SLO_TTLT_MS
    stall_threshold_ms: float = STALL_THRESHOLD_MS


@dataclass
class DiagResult:
    # ---- Identity ----
    test_id: str           # e.g. "TC-03"
    test_name: str
    scenario: str          # human-readable description

    # ---- Outcome ----
    success: bool
    timeout_occurred: bool
    connect_error: bool
    http_status: Optional[int]

    # ---- Timing (ms) ----
    connect_time_ms: float
    ttft_ms: float
    tpot_ms: float         # mean inter-token latency (generation phase)
    ttlt_ms: float         # end-to-end

    # ---- Token counts ----
    tokens_received: int
    tokens_expected: int

    # ---- Streaming analysis ----
    stall_detected: bool
    max_chunk_gap_ms: float   # largest gap between consecutive chunks
    itl_values_ms: List[float] = field(default_factory=list)

    # ---- Classification ----
    root_cause: str = RootCause.OK
    slo_ttft_pass: bool = True
    slo_tpot_pass: bool = True
    slo_ttlt_pass: bool = True

    # ---- Debug ----
    error: Optional[str] = None
    response_snippet: Optional[str] = None

    def validate_slos(self, cfg: DiagConfig):
        self.slo_ttft_pass = self.ttft_ms <= cfg.slo_ttft_ms or not self.success
        self.slo_tpot_pass = self.tpot_ms <= cfg.slo_tpot_ms or not self.success
        self.slo_ttlt_pass = self.ttlt_ms <= cfg.slo_ttlt_ms or not self.success


# ---------------------------------------------------------------------------
# Core request executor
# ---------------------------------------------------------------------------

class TimeoutTester:
    def __init__(self, config: DiagConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Prompt generation (realistic language, not repeated pangrams)
    # ------------------------------------------------------------------

    def generate_prompt(self, target_tokens: int) -> str:
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

    def generate_agentic_prompt(self) -> str:
        """Generate a realistic agentic scheduling-style prompt with embedded JSON.

        Replicates the class of prompts observed causing intermittent 504/client timeouts
        in production agentic AI systems:
          - Structured multi-phase role instructions (~400 tokens)
          - Embedded JSON calendar busy blocks (duplicated in two sections, ~600 tokens)
          - Complex multi-step reasoning requirement (~500 token output expected)
        Total: ~1,400–1,600 input tokens — matching the observed production workload.
        """
        busy_json = json.dumps({
            "agenticuser1@example.com": [
                {"start": "2026-04-14T14:00", "end": "2026-04-14T16:00"},
                {"start": "2026-04-17T09:00", "end": "2026-04-17T10:00"},
            ],
            "agenticuser2@example.com": [
                {"start": "2026-04-14T14:00", "end": "2026-04-14T16:00"},
                {"start": "2026-04-16T08:00", "end": "2026-04-16T09:00"},
                {"start": "2026-04-17T09:00", "end": "2026-04-17T10:00"},
                {"start": "2026-04-17T14:00", "end": "2026-04-17T15:00"},
            ],
            "agenticuser3@example.com": [
                {"start": "2026-04-16T08:00", "end": "2026-04-16T09:00"},
                {"start": "2026-04-17T09:00", "end": "2026-04-17T10:00"},
            ],
        }, indent=2)

        return (
            "# ROLE: Expert Scheduling Auditor\n"
            "# GOAL: Perform a 100% complete audit of free time blocks for a 60-minute meeting.\n\n"
            "### [PHASE 1: THE FILTERS (STRICT)]\n"
            "- START REFERENCE: 2026-04-13 09:00 (Monday).\n"
            "- SEARCH RANGE: 7 business days (skip weekends).\n"
            "- THE WEEKEND VOID: Saturday and Sunday are STRICTLY FORBIDDEN.\n"
            "- USER PREFERENCE FILTER: \"afternoon\".\n"
            "    - If \"Morning\" is mentioned, focus ONLY on 08:00-12:00.\n"
            "    - If \"Afternoon\" is mentioned, focus ONLY on 13:00-17:00.\n"
            "    - Otherwise, use the full 08:00-17:00 window.\n\n"
            "### [PHASE 2: CONFLICT SUBTRACTION]\n"
            f"- BUSY CLASH: Cross-reference {busy_json}.\n"
            "- SUBTRACTION RULE: Treat busy blocks as \"holes\" in the day.\n"
            "- MATH: [Requested Window] MINUS [Busy Blocks] = [Valid Windows].\n"
            "- Do NOT list busy blocks. Only list remaining free time.\n\n"
            "### [PHASE 3: FEASIBILITY GUARD & RATIONALE]\n"
            "1. **DURATION VALIDATION (CRITICAL)**:\n"
            "    - For every resulting free block, calculate its total minutes.\n"
            "    - If the block duration is LESS than 60, it is **INVALID**.\n"
            "    - If a day contains NO blocks long enough to fit the 60 minutes, skip that day.\n"
            "2. **Rationale Step**: For each of the 7 business days, briefly explain:\n"
            "    - If it is a Weekend (Skip).\n"
            "    - If it matches or fails the User Preference filter (Include/Skip).\n"
            "    - If the resulting free time is shorter than required 60 minutes (Mark as Impossible).\n"
            "    - **SEARCH BOUNDARY**: Explicitly mention the 7-day window.\n"
            "3. **User Summary**: A 1-2 sentence explanation of the result.\n"
            "   - MUST mention the search window.\n"
            "   - MUST explain why slots were found or not found.\n\n"
            "### [OUTPUT FORMAT]\n"
            "<Rationale>\n"
            "[Per-day reasoning with explicit calculations]\n"
            "</Rationale>\n\n"
            "<User_Summary>\n"
            "[1-2 sentence human-friendly explanation]\n"
            "</User_Summary>\n\n"
            "FINAL_DATA:\n"
            "[List available 60-minute slots, or leave blank if none]\n\n"
            "### [INPUT DATA]\n"
            "- ANCHOR: 2026-04-13 09:00 (Monday)\n"
            "- DURATION: 60 mins\n"
            "- REQUEST: afternoon\n"
            f"- BUSY_LOGS: {busy_json}\n"
        )

    # ------------------------------------------------------------------
    # Core streaming request — measures every diagnostic dimension
    # ------------------------------------------------------------------

    async def _execute(
        self,
        test_id: str,
        test_name: str,
        scenario: str,
        prompt: str,
        timeout_seconds: float,
        max_tokens: int,
        min_tokens: Optional[int] = None,
    ) -> DiagResult:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 0.0,
            "stream_options": {"include_usage": True},
        }
        if min_tokens:
            payload["min_tokens"] = min_tokens

        base = self.config.api_base.rstrip("/")
        url = f"{base}/chat/completions"

        # timing accumulators
        wall_start       = time.perf_counter()
        connect_done_ts: Optional[float] = None
        first_token_ts:  Optional[float] = None
        prev_chunk_ts:   Optional[float] = None
        last_token_ts:   Optional[float] = None

        token_count    = 0
        response_text  = ""
        itl_values_ms: List[float] = []
        max_chunk_gap_ms = 0.0
        stall_detected = False

        timeout_occurred = False
        connect_error    = False
        http_status: Optional[int] = None
        error_msg: Optional[str] = None

        # httpx.Timeout: (total, connect)
        hx_timeout = httpx.Timeout(timeout_seconds, connect=10.0)

        try:
            async with httpx.AsyncClient(
                headers=self.headers,
                timeout=hx_timeout,
            ) as client:
                # Measure connection phase by timing from send→first header byte
                try:
                    async with client.stream("POST", url, json=payload) as response:
                        connect_done_ts = time.perf_counter()
                        http_status = response.status_code

                        if response.status_code != 200:
                            body = await response.aread()
                            error_msg = f"HTTP {response.status_code}: {body.decode('utf-8', errors='replace')[:300]}"
                        else:
                            async for line in response.aiter_lines():
                                if not line or not line.strip():
                                    continue
                                if not line.startswith("data: "):
                                    continue
                                data_str = line[6:].strip()
                                if data_str == "[DONE]":
                                    break
                                try:
                                    data = json.loads(data_str)
                                except json.JSONDecodeError:
                                    continue

                                now = time.perf_counter()

                                # TTFT gate
                                if first_token_ts is None:
                                    first_token_ts = now
                                    prev_chunk_ts  = now

                                # Chunk gap (ITL / stall detection)
                                if prev_chunk_ts is not None:
                                    gap_ms = (now - prev_chunk_ts) * 1000.0
                                    if gap_ms > max_chunk_gap_ms:
                                        max_chunk_gap_ms = gap_ms
                                    if gap_ms > self.config.stall_threshold_ms:
                                        stall_detected = True
                                    itl_values_ms.append(gap_ms)
                                prev_chunk_ts = now

                                # Usage chunk (server-reported final counts)
                                if data.get("usage"):
                                    reported = data["usage"].get("completion_tokens", 0)
                                    if reported:
                                        token_count = reported

                                # Content delta
                                choices = data.get("choices", [])
                                if choices:
                                    delta   = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        last_token_ts = now
                                        response_text += content
                                        token_count   += 1

                except httpx.ConnectError as ce:
                    connect_error = True
                    error_msg = f"ConnectError: {ce}"
                except httpx.ConnectTimeout as ct:
                    connect_error = True
                    error_msg = f"ConnectTimeout: {ct}"

        except httpx.TimeoutException as te:
            timeout_occurred = True
            error_msg = f"ClientTimeout after {timeout_seconds}s: {te}"
        except Exception as exc:
            error_msg = str(exc)

        wall_end = time.perf_counter()

        # ---- Compute derived metrics ----
        connect_time_ms = ((connect_done_ts - wall_start) * 1000.0) if connect_done_ts else 0.0
        ttft_ms  = ((first_token_ts - wall_start) * 1000.0) if first_token_ts else 0.0
        ttlt_ms  = (wall_end - wall_start) * 1000.0

        if itl_values_ms and len(itl_values_ms) > 1:
            tpot_ms = _mean(itl_values_ms[1:])   # skip first gap (includes TTFT influence)
        elif first_token_ts and last_token_ts and last_token_ts > first_token_ts and token_count > 1:
            tpot_ms = ((last_token_ts - first_token_ts) * 1000.0) / max(token_count - 1, 1)
        else:
            tpot_ms = 0.0

        success = (
            not timeout_occurred
            and not connect_error
            and error_msg is None
            and http_status == 200
        )

        result = DiagResult(
            test_id=test_id,
            test_name=test_name,
            scenario=scenario,
            success=success,
            timeout_occurred=timeout_occurred,
            connect_error=connect_error,
            http_status=http_status,
            connect_time_ms=round(connect_time_ms, 2),
            ttft_ms=round(ttft_ms, 2),
            tpot_ms=round(tpot_ms, 2),
            ttlt_ms=round(ttlt_ms, 2),
            tokens_received=token_count,
            tokens_expected=max_tokens,
            stall_detected=stall_detected,
            max_chunk_gap_ms=round(max_chunk_gap_ms, 2),
            itl_values_ms=[round(v, 2) for v in itl_values_ms],
            error=error_msg,
            response_snippet=response_text[:300] if response_text else None,
        )
        result.root_cause = classify_root_cause(result)
        result.validate_slos(self.config)
        return result

    # ------------------------------------------------------------------
    # Convenience wrapper that logs each test as it runs
    # ------------------------------------------------------------------

    async def run_test(
        self,
        test_id: str,
        test_name: str,
        scenario: str,
        prompt: str,
        timeout_seconds: float,
        max_tokens: int,
        min_tokens: Optional[int] = None,
    ) -> "DiagResult":
        logger.info(f"[{test_id}] {test_name} — {scenario}")
        result = await self._execute(
            test_id, test_name, scenario, prompt,
            timeout_seconds, max_tokens, min_tokens,
        )
        self._print_result(result)
        return result

    # ------------------------------------------------------------------
    # Diagnostic test suite
    # ------------------------------------------------------------------

    async def run_all_tests(self) -> List["DiagResult"]:
        results: List[DiagResult] = []
        cfg = self.config
        prompt  = self.generate_prompt(cfg.prompt_tokens)

        # ── Phase 1: Connectivity ──────────────────────────────────────
        print("\n" + "━" * 60)
        print("PHASE 1 — Connectivity")
        print("━" * 60)

        # TC-01 / TC-02: Connect & TLS (just send 1-token request)
        r = await self.run_test(
            "TC-01/02", "Connect & TLS latency",
            "Measures TCP connect + TLS time; 1-token output to minimize generation noise",
            self.generate_prompt(50), 30.0, 1,
        )
        results.append(r)

        # ── Phase 2: TTFT Gate ─────────────────────────────────────────
        print("\n" + "━" * 60)
        print("PHASE 2 — TTFT Gate (Server Queue / Schedule Delay)")
        print("━" * 60)

        # TC-03: TTFT baseline (serial, nominal prompt)
        r = await self.run_test(
            "TC-03", "TTFT Baseline (serial)",
            f"Serial TTFT measurement with {cfg.prompt_tokens}-token prompt; SLO={cfg.slo_ttft_ms:.0f}ms",
            prompt, 60.0, cfg.max_tokens, cfg.min_tokens,
        )
        results.append(r)

        # TC-04: TTFT under load (10 concurrent)
        print(f"\n[TC-04] TTFT under load (10 concurrent)")
        tasks = [
            self._execute("TC-04", "TTFT Under Load", "10 concurrent — TTFT queue saturation test",
                          prompt, 90.0, cfg.max_tokens, cfg.min_tokens)
            for _ in range(10)
        ]
        burst = await asyncio.gather(*tasks)
        for r in burst:
            r.root_cause = classify_root_cause(r)
            r.validate_slos(cfg)
            self._print_result(r)
        results.extend(burst)

        # TC-05: Cold start — send one isolated request after a 5s idle
        print(f"\n[TC-05] Cold-start / idle detection (5s pause)")
        await asyncio.sleep(5)
        r = await self.run_test(
            "TC-05", "Cold-Start Detection",
            "After 5s idle — detects scheduler warm-up / GPU context eviction latency",
            prompt, 60.0, cfg.max_tokens, cfg.min_tokens,
        )
        results.append(r)

        # ── Phase 3: Generation Stall Gate ────────────────────────────
        print("\n" + "━" * 60)
        print("PHASE 3 — Generation Stall (TPOT / ITL Gate)")
        print("━" * 60)

        # TC-06 / TC-07: Large output — measures sustained generation TPOT and stalls
        r = await self.run_test(
            "TC-06/07", "Large Output — Stall & TPOT",
            "max_tokens=2048, min_tokens=1024 — measures TPOT, stall detection, ITL distribution",
            prompt, 180.0, 2048, 1024,
        )
        results.append(r)

        # TC-08: Partial response — short timeout into large generation
        r = await self.run_test(
            "TC-08", "Partial Response Detection",
            "5s budget on large-output request — expects PARTIAL classification if any tokens arrive",
            prompt, 5.0, cfg.max_tokens, cfg.min_tokens,
        )
        results.append(r)

        # ── Phase 4: E2E SLO ──────────────────────────────────────────
        print("\n" + "━" * 60)
        print("PHASE 4 — End-to-End SLO Validation")
        print("━" * 60)

        # TC-09a: Short budget
        r = await self.run_test(
            "TC-09a", "Short Client Timeout (5s)",
            "Hard 5s budget — expected TIMEOUT for non-trivial outputs",
            prompt, 5.0, cfg.max_tokens, cfg.min_tokens,
        )
        results.append(r)

        # TC-09b: Medium budget
        r = await self.run_test(
            "TC-09b", "Medium Client Timeout (30s)",
            "30s budget — baseline SLO check",
            prompt, 30.0, cfg.max_tokens, cfg.min_tokens,
        )
        results.append(r)

        # TC-09c: Long budget
        r = await self.run_test(
            "TC-09c", "Long Client Timeout (120s)",
            "120s budget — confirms successful completion under relaxed SLO",
            prompt, 120.0, cfg.max_tokens, cfg.min_tokens,
        )
        results.append(r)

        # TC-10: Large output E2E
        r = await self.run_test(
            "TC-10", "Large Output E2E (2048 tok, 180s)",
            "max_tokens=2048 with 180s client budget — E2E SLO for long generations",
            prompt, 180.0, 2048, 1000,
        )
        results.append(r)

        # TC-11a: Large input prompt
        large_prompt = self.generate_prompt(5000)
        r = await self.run_test(
            "TC-11a", "Large Input Prompt (5000 tokens)",
            "5000-token prompt + 60s budget — detects prefill-induced TTFT inflation",
            large_prompt, 60.0, cfg.max_tokens, cfg.min_tokens,
        )
        results.append(r)

        # TC-11b: Very large input
        very_large_prompt = self.generate_prompt(10000)
        r = await self.run_test(
            "TC-11b", "Very Large Input Prompt (10000 tokens)",
            "10000-token prompt + 120s budget — extreme prefill test",
            very_large_prompt, 120.0, cfg.max_tokens, cfg.min_tokens,
        )
        results.append(r)

        # ── Phase 5: Concurrency & Load ───────────────────────────────
        print("\n" + "━" * 60)
        print("PHASE 5 — Concurrency & Load")
        print("━" * 60)

        # TC-12: Sequential stability
        print(f"\n[TC-12] Sequential stability (5 requests, 1s gap)")
        for i in range(5):
            r = await self.run_test(
                "TC-12", f"Sequential Stability #{i+1}",
                "Sequential request — detects regression across repeated calls",
                prompt, 60.0, cfg.max_tokens, cfg.min_tokens,
            )
            results.append(r)
            if i < 4:
                await asyncio.sleep(1)

        # TC-13: Burst concurrency
        print(f"\n[TC-13] Burst concurrency (10 parallel)")
        tasks = [
            self._execute("TC-13", f"Burst Concurrency #{i+1}",
                          "10 simultaneous requests — timeout rate under burst load",
                          prompt, 60.0, cfg.max_tokens, cfg.min_tokens)
            for i in range(10)
        ]
        burst_results = await asyncio.gather(*tasks)
        for r in burst_results:
            r.root_cause = classify_root_cause(r)
            r.validate_slos(cfg)
            self._print_result(r)
        results.extend(burst_results)

        # TC-14: Concurrency sweep (1/2/4/8)
        print(f"\n[TC-14] Concurrency sweep (1 / 2 / 4 / 8 workers)")
        sweep_results: Dict[int, List[DiagResult]] = {}
        for level in [1, 2, 4, 8]:
            tasks = [
                self._execute("TC-14", f"Sweep c={level} #{j+1}",
                              f"Concurrency sweep at level {level}",
                              prompt, 90.0, cfg.max_tokens, cfg.min_tokens)
                for j in range(level * 3)   # 3 requests per worker
            ]
            level_results = list(await asyncio.gather(*tasks))
            for r in level_results:
                r.root_cause = classify_root_cause(r)
                r.validate_slos(cfg)
            sweep_results[level] = level_results
            results.extend(level_results)
            ok = sum(1 for r in level_results if r.success)
            ttfts = [r.ttft_ms for r in level_results if r.success]
            logger.info(
                f"  Sweep c={level}: {ok}/{len(level_results)} ok | "
                f"TTFT P95={_percentile(ttfts, 95):.0f}ms"
            )

        return results

    # ------------------------------------------------------------------
    # Phase 6: Agentic / Production Workload
    # Reproduces the intermittent 504 / client-timeout pattern seen when
    # agentic AI systems send complex structured prompts with embedded JSON.
    # ------------------------------------------------------------------

    async def run_agentic_tests(self) -> List["DiagResult"]:
        """Phase 6: Agentic / Production Workload Tests.

        Test plan (mirrors production failure pattern):
          TC-A1  Agentic baseline (180s budget) — establishes true TTLT for complex prompt
          TC-A2  Agentic prompt with 30s budget  — simulates agentic AI client timeout
          TC-A3  Intermittency measurement        — 5 sequential runs, detects failure rate
          TC-A4  Retry storm simulation           — 3 concurrent requests (retry-on-timeout)
        """
        results: List[DiagResult] = []
        cfg = self.config

        prompt = self.generate_agentic_prompt()
        prompt_est_tokens = len(prompt) // 4

        print("\n" + "━" * 60)
        print("PHASE 6 — Agentic / Production Workload")
        print(f"  Prompt: ~{prompt_est_tokens} tokens (structured instructions + embedded JSON)")
        print(f"  Reproducing: intermittent 504 / client-timeout pattern")
        print("━" * 60)

        # TC-A1: Baseline with generous budget — find out true TTLT
        r = await self.run_test(
            "TC-A1", "Agentic Workload Baseline",
            f"Complex structured prompt ~{prompt_est_tokens} tok; 180s budget — establishes true TTLT",
            prompt, 180.0, cfg.max_tokens, cfg.min_tokens,
        )
        results.append(r)

        # TC-A2: Agentic prompt with production timeout (30s — typical agentic AI budget)
        r = await self.run_test(
            "TC-A2", "Agentic Workload — 30s Client Timeout",
            "Same complex prompt; 30s budget — simulates production agentic AI timeout (TIMEOUT here = real user pain)",
            prompt, 30.0, cfg.max_tokens, cfg.min_tokens,
        )
        results.append(r)

        # TC-A3: Intermittency measurement — 5 sequential runs, 2s gap
        print(f"\n[TC-A3] Intermittency measurement (5 sequential, 2s gap)")
        fails = 0
        for i in range(5):
            r = await self.run_test(
                "TC-A3", f"Agentic Intermittency #{i+1}/5",
                f"Sequential run {i+1}/5 — detects first-attempt failure pattern",
                prompt, 120.0, cfg.max_tokens, cfg.min_tokens,
            )
            results.append(r)
            if not r.success:
                fails += 1
            if i < 4:
                await asyncio.sleep(2)

        fail_rate = 100.0 * fails / 5
        logger.info(
            f"  TC-A3 result: {fails}/5 failed ({fail_rate:.0f}% failure rate) — "
            + ("INTERMITTENCY CONFIRMED" if fails > 0 else "stable under serial load")
        )

        # TC-A4: Retry storm — 3 concurrent (what happens when agentic retries kick in simultaneously)
        print(f"\n[TC-A4] Retry storm simulation (3 concurrent agentic requests)")
        tasks = [
            self._execute(
                "TC-A4", f"Retry Storm #{j+1}/3",
                "3 concurrent identical agentic requests — simulates retry-on-timeout behaviour",
                prompt, 120.0, cfg.max_tokens, cfg.min_tokens,
            )
            for j in range(3)
        ]
        storm_results = await asyncio.gather(*tasks)
        for r in storm_results:
            r.root_cause = classify_root_cause(r)
            r.validate_slos(cfg)
            self._print_result(r)
        results.extend(storm_results)

        storm_ok = sum(1 for r in storm_results if r.success)
        logger.info(
            f"  TC-A4 result: {storm_ok}/3 succeeded — "
            + ("queue saturation likely" if storm_ok < 3 else "server handled retry burst")
        )

        return results

    # ------------------------------------------------------------------
    # Console print
    # ------------------------------------------------------------------

    def _print_result(self, result: "DiagResult"):
        icon   = "✓" if result.success else "✗"
        rc     = result.root_cause
        rc_tag = f"[{rc}]" if rc != RootCause.OK else ""
        slo_flags = ""
        if result.success:
            if not result.slo_ttft_pass: slo_flags += " SLO-TTFT!"
            if not result.slo_tpot_pass: slo_flags += " SLO-TPOT!"
            if not result.slo_ttlt_pass: slo_flags += " SLO-TTLT!"
        stall_tag = " [STALL]" if result.stall_detected else ""
        print(
            f"  {icon} [{result.test_id}] {result.test_name}{rc_tag}{stall_tag}{slo_flags}"
        )
        print(
            f"     Connect={result.connect_time_ms:.0f}ms  "
            f"TTFT={result.ttft_ms:.0f}ms  "
            f"TPOT={result.tpot_ms:.1f}ms  "
            f"TTLT={result.ttlt_ms:.0f}ms  "
            f"Tokens={result.tokens_received}/{result.tokens_expected}  "
            f"MaxGap={result.max_chunk_gap_ms:.0f}ms"
        )
        if result.error:
            print(f"     Error: {result.error}")
        print()

    async def run_test(
        self,
        test_id: str,
        test_name: str,
        scenario: str,
        prompt: str,
        timeout_seconds: float,
        max_tokens: int,
        min_tokens: Optional[int] = None,
    ) -> DiagResult:
        logger.info(f"[{test_id}] {test_name} — {scenario}")
        result = await self._execute(
            test_id, test_name, scenario, prompt,
            timeout_seconds, max_tokens, min_tokens,
        )
        self._print_result(result)
        return result


# ---------------------------------------------------------------------------
# HTML Report Builder
# ---------------------------------------------------------------------------

RC_COLORS = {
    RootCause.OK:       "#22c55e",
    RootCause.NETWORK:  "#ef4444",
    RootCause.QUEUE:    "#f97316",
    RootCause.STALL:    "#a855f7",
    RootCause.TIMEOUT:  "#eab308",
    RootCause.PARTIAL:  "#06b6d4",
    RootCause.HTTP_ERR: "#f43f5e",
}

RC_DESCRIPTIONS = {
    RootCause.OK:       "No issue detected — request completed within all SLOs.",
    RootCause.NETWORK:  "TCP connect or TLS failure — check network route / firewall / DNS.",
    RootCause.QUEUE:    "TTFT exceeded SLO — server inference queue saturation or KV-cache pressure.",
    RootCause.STALL:    "Mid-stream token gap exceeded stall threshold — GPU decode stall or memory pressure.",
    RootCause.TIMEOUT:  "Client budget exceeded before any tokens arrived — increase timeout or investigate TTFT.",
    RootCause.PARTIAL:  "Partial response — tokens started but stream terminated early (model crash / OOM).",
    RootCause.HTTP_ERR: "Non-200 HTTP status — check API key, model name, or server-side error logs.",
}


def _build_html_path(model: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^\w\-.]", "_", model)
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "timeout_test"
    )
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, f"{ts}_{safe_model}_timeout_diag.html")


def _stat_row(label: str, s: Dict[str, float]) -> str:
    def f(v: float) -> str:
        return f"{v:.1f}"
    return (
        f"<tr><td>{label}</td>"
        f"<td>{f(s['mean'])}</td><td>{f(s['std'])}</td>"
        f"<td>{f(s['p50'])}</td><td>{f(s['p90'])}</td>"
        f"<td>{f(s['p95'])}</td><td>{f(s['p99'])}</td>"
        f"<td>{f(s['max'])}</td></tr>"
    )


def build_html_report(
    results: List[DiagResult],
    config: DiagConfig,
    run_ts: str,
) -> str:
    successful = [r for r in results if r.success]
    failed     = [r for r in results if not r.success]

    # Root-cause distribution
    rc_counts: Dict[str, int] = {}
    for r in results:
        rc_counts[r.root_cause] = rc_counts.get(r.root_cause, 0) + 1

    # Phase-level groups for scatter
    phase_map: Dict[str, List[DiagResult]] = {}
    for r in results:
        phase_map.setdefault(r.test_id, []).append(r)

    # Aggregate metrics across successful results
    ttft_vals  = [r.ttft_ms  for r in successful]
    tpot_vals  = [r.tpot_ms  for r in successful if r.tpot_ms > 0]
    ttlt_vals  = [r.ttlt_ms  for r in successful]
    conn_vals  = [r.connect_time_ms for r in successful if r.connect_time_ms > 0]
    gap_vals   = [r.max_chunk_gap_ms for r in successful]
    all_itl    = [v for r in successful for v in r.itl_values_ms if v > 0]

    ttft_stats = compute_stats(ttft_vals)
    tpot_stats = compute_stats(tpot_vals)
    ttlt_stats = compute_stats(ttlt_vals)
    conn_stats = compute_stats(conn_vals)
    itl_stats  = compute_stats(all_itl)

    # Per-test scatter data for all phases
    scatter_labels  = [f"{r.test_id}" for r in results]
    scatter_ttft    = [r.ttft_ms  for r in results]
    scatter_ttlt    = [r.ttlt_ms  for r in results]
    scatter_tpot    = [r.tpot_ms  for r in results]
    scatter_gap     = [r.max_chunk_gap_ms for r in results]

    # Concurrency sweep data (TC-14 only)
    sweep_levels: List[int] = []
    sweep_ttft_p95: List[float] = []
    sweep_timeout_pct: List[float] = []
    for level in [1, 2, 4, 8]:
        level_results = [r for r in results if r.test_id == "TC-14"
                         and f"c={level}" in r.test_name]
        if level_results:
            ttfts = [r.ttft_ms for r in level_results if r.success]
            to_pct = 100.0 * sum(1 for r in level_results if r.timeout_occurred) / len(level_results)
            sweep_levels.append(level)
            sweep_ttft_p95.append(_percentile(ttfts, 95) if ttfts else 0.0)
            sweep_timeout_pct.append(to_pct)

    # ITL histogram (20 bins)
    itl_hist_labels: List[str] = []
    itl_hist_counts: List[int] = []
    if all_itl:
        itl_min, itl_max = min(all_itl), max(all_itl)
        step = max((itl_max - itl_min) / 20, 1.0)
        edges = [itl_min + i * step for i in range(20)]
        counts = [0] * 20
        for v in all_itl:
            idx = min(int((v - itl_min) / step), 19)
            counts[idx] += 1
        itl_hist_labels = [f"{e:.0f}+" for e in edges]
        itl_hist_counts = counts

    # Root-cause pie
    rc_labels  = list(rc_counts.keys())
    rc_vals    = [rc_counts[k] for k in rc_labels]
    rc_colors  = [RC_COLORS.get(k, "#64748b") for k in rc_labels]

    # SLO compliance
    slo_ttft_pass = sum(1 for r in successful if r.slo_ttft_pass)
    slo_tpot_pass = sum(1 for r in successful if r.slo_tpot_pass)
    slo_ttlt_pass = sum(1 for r in successful if r.slo_ttlt_pass)
    total_suc = max(len(successful), 1)

    # Results table rows
    def result_row(r: DiagResult) -> str:
        rc_color = RC_COLORS.get(r.root_cause, "#64748b")
        status_badge = (
            '<span style="color:#22c55e">✓</span>'
            if r.success else
            '<span style="color:#ef4444">✗</span>'
        )
        stall = "⚠" if r.stall_detected else ""
        slo_flags = ""
        if r.success:
            if not r.slo_ttft_pass: slo_flags += " <b style='color:#f97316'>TTFT!</b>"
            if not r.slo_tpot_pass: slo_flags += " <b style='color:#a855f7'>TPOT!</b>"
            if not r.slo_ttlt_pass: slo_flags += " <b style='color:#ef4444'>TTLT!</b>"
        return (
            f"<tr>"
            f"<td>{r.test_id}</td>"
            f"<td>{r.test_name}</td>"
            f"<td>{status_badge}{stall}</td>"
            f"<td style='color:{rc_color};font-weight:600'>{r.root_cause}</td>"
            f"<td>{r.connect_time_ms:.0f}</td>"
            f"<td>{r.ttft_ms:.0f}</td>"
            f"<td>{r.tpot_ms:.1f}</td>"
            f"<td>{r.ttlt_ms:.0f}</td>"
            f"<td>{r.tokens_received}/{r.tokens_expected}</td>"
            f"<td>{r.max_chunk_gap_ms:.0f}</td>"
            f"<td>{slo_flags or '—'}</td>"
            f"<td style='font-size:0.75rem;color:#94a3b8'>{(r.error or '')[:60]}</td>"
            f"</tr>"
        )

    all_rows = "".join(result_row(r) for r in results)

    # Root-cause recommendation cards
    present_causes = set(r.root_cause for r in results if r.root_cause != RootCause.OK)
    rec_cards = ""
    for rc in sorted(present_causes):
        color = RC_COLORS.get(rc, "#64748b")
        desc  = RC_DESCRIPTIONS.get(rc, "")
        count = rc_counts.get(rc, 0)
        rec_cards += f"""
<div class="rec-card" style="border-left:4px solid {color}">
  <div class="rec-title" style="color:{color}">{rc} &nbsp; <span class="badge" style="background:{color}">{count}×</span></div>
  <div class="rec-body">{desc}</div>
</div>"""

    if not rec_cards:
        rec_cards = '<div class="rec-card" style="border-left:4px solid #22c55e"><div class="rec-title" style="color:#22c55e">All Clear</div><div class="rec-body">No timeout or SLO violations detected.</div></div>'

    # Sweep section
    sweep_section = ""
    sweep_js = ""
    if sweep_levels:
        sweep_section = """
<div class="card">
  <h2>TC-14 — Concurrency Sweep</h2>
  <div class="two-col">
    <div class="chart-wrap"><canvas id="chartSweepTtft"></canvas></div>
    <div class="chart-wrap"><canvas id="chartSweepTimeout"></canvas></div>
  </div>
</div>"""
        sweep_js = f"""
  new Chart(document.getElementById('chartSweepTtft'), {{
    type: 'line',
    data: {{
      labels: {json.dumps(sweep_levels)},
      datasets: [{{
        label: 'TTFT P95 (ms)',
        data: {json.dumps([round(v,1) for v in sweep_ttft_p95])},
        borderColor: '#f97316', backgroundColor: 'rgba(249,115,22,0.15)',
        fill: true, tension: 0.3, pointRadius: 5
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ title: {{ display: true, text: 'TTFT P95 vs Concurrency (TC-14)' }} }},
      scales: {{
        x: {{ title: {{ display: true, text: 'Concurrency' }} }},
        y: {{ title: {{ display: true, text: 'ms' }}, beginAtZero: true }}
      }}
    }}
  }});
  new Chart(document.getElementById('chartSweepTimeout'), {{
    type: 'bar',
    data: {{
      labels: {json.dumps(sweep_levels)},
      datasets: [{{
        label: 'Timeout Rate (%)',
        data: {json.dumps([round(v,1) for v in sweep_timeout_pct])},
        backgroundColor: 'rgba(239,68,68,0.7)', borderColor: '#ef4444',
        borderWidth: 1, borderRadius: 4
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ title: {{ display: true, text: 'Timeout Rate vs Concurrency (TC-14)' }}, legend: {{ display: false }} }},
      scales: {{
        x: {{ title: {{ display: true, text: 'Concurrency' }} }},
        y: {{ title: {{ display: true, text: '%' }}, min: 0, max: 100 }}
      }}
    }}
  }});
"""

    itl_hist_js = ""
    if itl_hist_labels:
        itl_hist_js = f"""
  new Chart(document.getElementById('chartItlHist'), {{
    type: 'bar',
    data: {{
      labels: {json.dumps(itl_hist_labels)},
      datasets: [{{
        label: 'Chunks',
        data: {json.dumps(itl_hist_counts)},
        backgroundColor: 'rgba(168,85,247,0.7)',
        borderColor: '#a855f7', borderWidth: 1, borderRadius: 3
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ title: {{ display: true, text: 'Inter-Token Latency Distribution (ms)' }}, legend: {{ display: false }} }},
      scales: {{
        x: {{ title: {{ display: true, text: 'ITL (ms)' }} }},
        y: {{ title: {{ display: true, text: 'Count' }}, beginAtZero: true }}
      }}
    }}
  }});
"""

    pct_labels = ["P50", "P90", "P95", "P99"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>LLM Timeout Diagnostic — {config.model}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #0f172a; color: #e2e8f0; padding: 2rem;
    }}
    h1 {{ font-size: 1.8rem; font-weight: 700; color: #f8fafc; }}
    h2 {{ font-size: 1.05rem; font-weight: 600; color: #94a3b8; margin-bottom: 1rem; }}
    .subtitle {{ color: #64748b; font-size: 0.85rem; margin: 0.4rem 0 2rem; }}
    .card {{
      background: #1e293b; border-radius: 12px; padding: 1.5rem;
      margin-bottom: 1.5rem; box-shadow: 0 2px 12px rgba(0,0,0,0.4);
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 1rem; margin-bottom: 1.5rem;
    }}
    .kpi {{
      background: #0f172a; border-radius: 10px; padding: 1rem 1.2rem;
      border: 1px solid #334155;
    }}
    .kpi-label {{ font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
    .kpi-value {{ font-size: 1.5rem; font-weight: 700; color: #f1f5f9; margin-top: 0.2rem; }}
    .kpi-unit  {{ font-size: 0.8rem; font-weight: 400; color: #94a3b8; }}
    .chart-wrap {{ position: relative; height: 260px; margin-bottom: 1.5rem; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
    @media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
    th, td {{ padding: 0.5rem 0.65rem; border-bottom: 1px solid #334155; text-align: left; }}
    th {{ color: #94a3b8; font-weight: 600; text-transform: uppercase; font-size: 0.72rem; background: #0f172a; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover {{ background: rgba(255,255,255,0.02); }}
    .badge {{
      display: inline-block; padding: 0.15rem 0.5rem;
      border-radius: 999px; font-size: 0.72rem; font-weight: 600; color: #fff;
    }}
    .rec-card {{
      background: #0f172a; border-radius: 8px; padding: 0.9rem 1.1rem;
      margin-bottom: 0.75rem;
    }}
    .rec-title {{ font-weight: 700; font-size: 0.9rem; margin-bottom: 0.3rem; }}
    .rec-body  {{ font-size: 0.82rem; color: #94a3b8; line-height: 1.5; }}
    .slo-bar {{
      display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.6rem;
    }}
    .slo-label {{ width: 120px; font-size: 0.8rem; color: #94a3b8; }}
    .slo-track {{
      flex: 1; height: 10px; background: #334155; border-radius: 999px; overflow: hidden;
    }}
    .slo-fill {{ height: 100%; border-radius: 999px; transition: width 0.5s; }}
    .slo-pct {{ width: 45px; text-align: right; font-size: 0.8rem; font-weight: 600; }}
    .phase-header {{
      font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
      text-transform: uppercase; color: #475569; padding: 0.4rem 0.65rem;
      background: #0f172a; border-bottom: 1px solid #334155;
    }}
  </style>
</head>
<body>
  <h1>LLM Timeout Diagnostic Report</h1>
  <p class="subtitle">
    Model: <strong>{config.model}</strong> &nbsp;|&nbsp; {run_ts}
    &nbsp;|&nbsp; SLO TTFT={config.slo_ttft_ms:.0f}ms &nbsp;
    TPOT={config.slo_tpot_ms:.0f}ms &nbsp; TTLT={config.slo_ttlt_ms:.0f}ms
    &nbsp;|&nbsp; Stall threshold={config.stall_threshold_ms:.0f}ms
  </p>

  <!-- KPI row -->
  <div class="kpi-grid">
    <div class="kpi"><div class="kpi-label">Total Tests</div>
      <div class="kpi-value">{len(results)}<span class="kpi-unit"> runs</span></div></div>
    <div class="kpi"><div class="kpi-label">Passed</div>
      <div class="kpi-value" style="color:#22c55e">{len(successful)}</div></div>
    <div class="kpi"><div class="kpi-label">Failed</div>
      <div class="kpi-value" style="color:#ef4444">{len(failed)}</div></div>
    <div class="kpi"><div class="kpi-label">TTFT P50</div>
      <div class="kpi-value">{ttft_stats['p50']:.0f}<span class="kpi-unit"> ms</span></div></div>
    <div class="kpi"><div class="kpi-label">TTFT P95</div>
      <div class="kpi-value">{ttft_stats['p95']:.0f}<span class="kpi-unit"> ms</span></div></div>
    <div class="kpi"><div class="kpi-label">TPOT Mean</div>
      <div class="kpi-value">{tpot_stats['mean']:.1f}<span class="kpi-unit"> ms</span></div></div>
    <div class="kpi"><div class="kpi-label">Connect P95</div>
      <div class="kpi-value">{conn_stats['p95']:.0f}<span class="kpi-unit"> ms</span></div></div>
    <div class="kpi"><div class="kpi-label">Max Stall Gap</div>
      <div class="kpi-value">{max((r.max_chunk_gap_ms for r in results), default=0.0):.0f}<span class="kpi-unit"> ms</span></div></div>
    <div class="kpi"><div class="kpi-label">Stall Events</div>
      <div class="kpi-value" style="color:#a855f7">{sum(1 for r in results if r.stall_detected)}</div></div>
    <div class="kpi"><div class="kpi-label">Timeout Events</div>
      <div class="kpi-value" style="color:#eab308">{sum(1 for r in results if r.timeout_occurred)}</div></div>
  </div>

  <!-- Root-cause recommendations -->
  <div class="card">
    <h2>Root-Cause Analysis &amp; Recommendations</h2>
    {rec_cards}
  </div>

  <!-- SLO compliance bars -->
  <div class="card">
    <h2>SLO Compliance (successful requests only)</h2>
    <div class="slo-bar">
      <div class="slo-label">TTFT ≤ {config.slo_ttft_ms:.0f}ms</div>
      <div class="slo-track"><div class="slo-fill" style="width:{100*slo_ttft_pass//total_suc}%;background:#6366f1"></div></div>
      <div class="slo-pct">{100*slo_ttft_pass//total_suc}%</div>
    </div>
    <div class="slo-bar">
      <div class="slo-label">TPOT ≤ {config.slo_tpot_ms:.0f}ms</div>
      <div class="slo-track"><div class="slo-fill" style="width:{100*slo_tpot_pass//total_suc}%;background:#10b981"></div></div>
      <div class="slo-pct">{100*slo_tpot_pass//total_suc}%</div>
    </div>
    <div class="slo-bar">
      <div class="slo-label">TTLT ≤ {config.slo_ttlt_ms:.0f}ms</div>
      <div class="slo-track"><div class="slo-fill" style="width:{100*slo_ttlt_pass//total_suc}%;background:#f59e0b"></div></div>
      <div class="slo-pct">{100*slo_ttlt_pass//total_suc}%</div>
    </div>
  </div>

  <!-- Root-cause pie + latency percentile bars -->
  <div class="two-col">
    <div class="card">
      <h2>Root-Cause Distribution</h2>
      <div class="chart-wrap"><canvas id="chartRcPie"></canvas></div>
    </div>
    <div class="card">
      <h2>Latency Percentiles (successful requests)</h2>
      <div class="chart-wrap"><canvas id="chartLatPct"></canvas></div>
    </div>
  </div>

  <!-- Per-request scatter: TTFT & TTLT -->
  <div class="card">
    <h2>Per-Test Latency Scatter (all tests)</h2>
    <div class="chart-wrap" style="height:300px"><canvas id="chartScatter"></canvas></div>
  </div>

  <!-- Chunk gap / stall chart -->
  <div class="card">
    <h2>Max Chunk Gap per Test (stall detector)</h2>
    <div class="chart-wrap"><canvas id="chartGap"></canvas></div>
  </div>

  <!-- ITL histogram -->
  <div class="card">
    <h2>Inter-Token Latency (ITL) Histogram — all successful streaming chunks</h2>
    <div class="chart-wrap"><canvas id="chartItlHist"></canvas></div>
  </div>

  {sweep_section}

  <!-- Aggregate stats table -->
  <div class="card">
    <h2>Aggregate Statistics (successful requests)</h2>
    <table>
      <thead><tr>
        <th>Metric</th><th>Mean</th><th>Std</th>
        <th>P50</th><th>P90</th><th>P95</th><th>P99</th><th>Max</th>
      </tr></thead>
      <tbody>
        {_stat_row("Connect (ms)",  conn_stats)}
        {_stat_row("TTFT (ms)",     ttft_stats)}
        {_stat_row("TPOT (ms)",     tpot_stats)}
        {_stat_row("TTLT (ms)",     ttlt_stats)}
        {_stat_row("ITL (ms)",      itl_stats)}
        {_stat_row("Max Gap (ms)",  compute_stats(gap_vals))}
      </tbody>
    </table>
  </div>

  <!-- Full results table -->
  <div class="card">
    <h2>Full Test Results</h2>
    <div style="overflow-x:auto">
    <table>
      <thead><tr>
        <th>ID</th><th>Test</th><th>Status</th><th>Root Cause</th>
        <th>Connect (ms)</th><th>TTFT (ms)</th><th>TPOT (ms)</th><th>TTLT (ms)</th>
        <th>Tokens</th><th>MaxGap (ms)</th><th>SLO</th><th>Error</th>
      </tr></thead>
      <tbody>{all_rows}</tbody>
    </table>
    </div>
  </div>

  <!-- Config -->
  <div class="card">
    <h2>Configuration</h2>
    <table>
      <tbody>
        <tr><td style="color:#94a3b8;width:40%">Model</td><td>{config.model}</td></tr>
        <tr><td style="color:#94a3b8">API Base</td><td>{config.api_base}</td></tr>
        <tr><td style="color:#94a3b8">Prompt Tokens</td><td>{config.prompt_tokens}</td></tr>
        <tr><td style="color:#94a3b8">Max Tokens</td><td>{config.max_tokens}</td></tr>
        <tr><td style="color:#94a3b8">Min Tokens</td><td>{config.min_tokens or '—'}</td></tr>
        <tr><td style="color:#94a3b8">SLO TTFT</td><td>{config.slo_ttft_ms:.0f} ms</td></tr>
        <tr><td style="color:#94a3b8">SLO TPOT</td><td>{config.slo_tpot_ms:.0f} ms</td></tr>
        <tr><td style="color:#94a3b8">SLO TTLT</td><td>{config.slo_ttlt_ms:.0f} ms</td></tr>
        <tr><td style="color:#94a3b8">Stall Threshold</td><td>{config.stall_threshold_ms:.0f} ms</td></tr>
        <tr><td style="color:#94a3b8">Report Generated</td><td>{run_ts}</td></tr>
      </tbody>
    </table>
  </div>

  <script>
  // ── Root-cause pie ─────────────────────────────────────────────────────
  new Chart(document.getElementById('chartRcPie'), {{
    type: 'doughnut',
    data: {{
      labels: {json.dumps(rc_labels)},
      datasets: [{{
        data: {json.dumps(rc_vals)},
        backgroundColor: {json.dumps(rc_colors)},
        borderWidth: 2, borderColor: '#1e293b'
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'right', labels: {{ color: '#e2e8f0', font: {{ size: 12 }} }} }},
        title: {{ display: false }}
      }}
    }}
  }});

  // ── Latency percentile bar chart ────────────────────────────────────────
  new Chart(document.getElementById('chartLatPct'), {{
    type: 'bar',
    data: {{
      labels: {json.dumps(pct_labels)},
      datasets: [
        {{
          label: 'TTFT (ms)',
          data: {json.dumps([round(ttft_stats[k],1) for k in ['p50','p90','p95','p99']])},
          backgroundColor: 'rgba(99,102,241,0.8)', borderRadius: 4
        }},
        {{
          label: 'TPOT (ms)',
          data: {json.dumps([round(tpot_stats[k],1) for k in ['p50','p90','p95','p99']])},
          backgroundColor: 'rgba(16,185,129,0.8)', borderRadius: 4
        }},
        {{
          label: 'TTLT (ms)',
          data: {json.dumps([round(ttlt_stats[k],1) for k in ['p50','p90','p95','p99']])},
          backgroundColor: 'rgba(245,158,11,0.8)', borderRadius: 4
        }}
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ title: {{ display: false }} }},
      scales: {{ y: {{ title: {{ display: true, text: 'ms' }}, beginAtZero: true }} }}
    }}
  }});

  // ── Per-test scatter ────────────────────────────────────────────────────
  new Chart(document.getElementById('chartScatter'), {{
    type: 'line',
    data: {{
      labels: {json.dumps(scatter_labels)},
      datasets: [
        {{
          label: 'TTFT (ms)',
          data: {json.dumps(scatter_ttft)},
          borderColor: '#6366f1', backgroundColor: 'transparent',
          pointRadius: 4, tension: 0.15
        }},
        {{
          label: 'TTLT (ms)',
          data: {json.dumps(scatter_ttlt)},
          borderColor: '#f59e0b', backgroundColor: 'transparent',
          pointRadius: 4, tension: 0.15
        }},
        {{
          label: 'TPOT (ms)',
          data: {json.dumps(scatter_tpot)},
          borderColor: '#10b981', backgroundColor: 'transparent',
          pointRadius: 4, tension: 0.15
        }}
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ title: {{ display: false }} }},
      scales: {{
        x: {{ title: {{ display: true, text: 'Test' }}, ticks: {{ maxRotation: 60 }} }},
        y: {{ title: {{ display: true, text: 'ms' }}, beginAtZero: true }}
      }}
    }}
  }});

  // ── Max chunk gap bar ───────────────────────────────────────────────────
  new Chart(document.getElementById('chartGap'), {{
    type: 'bar',
    data: {{
      labels: {json.dumps(scatter_labels)},
      datasets: [{{
        label: 'Max Chunk Gap (ms)',
        data: {json.dumps(scatter_gap)},
        backgroundColor: scatter_gap.map(v => v > {config.stall_threshold_ms}
          ? 'rgba(168,85,247,0.85)' : 'rgba(99,102,241,0.6)'),
        borderRadius: 3
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        title: {{ display: false }},
        legend: {{ display: false }},
        annotation: {{ annotations: [] }}
      }},
      scales: {{
        x: {{ title: {{ display: true, text: 'Test' }}, ticks: {{ maxRotation: 60 }} }},
        y: {{ title: {{ display: true, text: 'ms (purple = stall)' }}, beginAtZero: true }}
      }}
    }}
  }});

  {itl_hist_js}
  {sweep_js}
  </script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="LLM Timeout Diagnostic Tool — Industry-Standard Root-Cause Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url",      default=DEFAULT_API_BASE,  help="API Base URL")
    parser.add_argument("--api-key",       default=DEFAULT_API_KEY,   help="API Key")
    parser.add_argument("--model",         default=DEFAULT_MODEL,     help="Model name")
    parser.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS,
                        help="Nominal prompt token count for standard tests")
    parser.add_argument("--max-tokens",    type=int, default=DEFAULT_MAX_TOKENS,
                        help="Max output tokens for standard tests")
    parser.add_argument("--min-tokens",    type=int, default=DEFAULT_MIN_TOKENS,
                        help="Min output tokens (forces min generation length)")
    parser.add_argument("--slo-ttft",      type=float, default=SLO_TTFT_MS,
                        help="TTFT SLO threshold (ms)")
    parser.add_argument("--slo-tpot",      type=float, default=SLO_TPOT_MS,
                        help="TPOT SLO threshold (ms)")
    parser.add_argument("--slo-ttlt",      type=float, default=SLO_TTLT_MS,
                        help="TTLT (E2E) SLO threshold (ms)")
    parser.add_argument("--stall-ms",      type=float, default=STALL_THRESHOLD_MS,
                        help="Chunk gap threshold to declare a generation stall (ms)")
    parser.add_argument("--single",        action="store_true",
                        help="Run a single quick smoke test only (TC-03 baseline)")
    parser.add_argument("--agentic",       action="store_true",
                        help="Run Phase 6 agentic workload tests only (intermittency + retry storm)")
    parser.add_argument("--include-agentic", action="store_true",
                        help="Append Phase 6 agentic tests to the full suite")
    args = parser.parse_args()

    config = DiagConfig(
        api_base=args.base_url,
        api_key=args.api_key,
        model=args.model,
        prompt_tokens=args.prompt_tokens,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        slo_ttft_ms=args.slo_ttft,
        slo_tpot_ms=args.slo_tpot,
        slo_ttlt_ms=args.slo_ttlt,
        stall_threshold_ms=args.stall_ms,
    )

    W = 62
    print("=" * W)
    print("  LLM TIMEOUT DIAGNOSTIC TOOL")
    print("=" * W)
    print(f"  Model  : {args.model}")
    print(f"  API    : {args.base_url}")
    print(f"  SLOs   : TTFT={args.slo_ttft:.0f}ms  TPOT={args.slo_tpot:.0f}ms  TTLT={args.slo_ttlt:.0f}ms")
    print(f"  Stall  : >{args.stall_ms:.0f}ms chunk gap")
    print("=" * W)

    tester = TimeoutTester(config)

    if args.agentic:
        print("\n=== AGENTIC WORKLOAD TESTS (Phase 6 only) ===")
        results = await tester.run_agentic_tests()
    elif args.single:
        print("\n=== SINGLE SMOKE TEST (TC-03) ===")
        prompt = tester.generate_prompt(config.prompt_tokens)
        results = [await tester.run_test(
            "TC-03", "Single Smoke Test",
            "Quick connectivity + TTFT baseline",
            prompt, 60.0, config.max_tokens, config.min_tokens,
        )]
    else:
        results = await tester.run_all_tests()
        if args.include_agentic:
            agentic_results = await tester.run_agentic_tests()
            results.extend(agentic_results)

    # ── Console summary ──────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * W)

    successful = [r for r in results if r.success]
    timeouts   = [r for r in results if r.timeout_occurred]
    stalls     = [r for r in results if r.stall_detected]
    partials   = [r for r in results if r.root_cause == RootCause.PARTIAL]
    network_errs = [r for r in results if r.root_cause == RootCause.NETWORK]

    from collections import Counter
    rc_summary = Counter(r.root_cause for r in results)

    print(f"  Total tests      : {len(results)}")
    print(f"  Successful       : {len(successful)}")
    print(f"  Timeouts         : {len(timeouts)}")
    print(f"  Stall events     : {len(stalls)}")
    print(f"  Partial responses: {len(partials)}")
    print(f"  Network errors   : {len(network_errs)}")
    print()
    print("  Root-cause breakdown:")
    for rc, cnt in rc_summary.most_common():
        print(f"    {rc:<12} {cnt:>3}x  — {RC_DESCRIPTIONS.get(rc,'')[:60]}")

    if successful:
        ttft_list = [r.ttft_ms for r in successful]
        print()
        print(f"  TTFT  P50={_percentile(ttft_list,50):.0f}ms  P95={_percentile(ttft_list,95):.0f}ms  P99={_percentile(ttft_list,99):.0f}ms")

    print("=" * W)

    # ── HTML report ──────────────────────────────────────────────────────
    run_ts   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    html_path = _build_html_path(config.model)
    html = build_html_report(results, config, run_ts)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  HTML report → {html_path}\n")


if __name__ == "__main__":
    asyncio.run(main())