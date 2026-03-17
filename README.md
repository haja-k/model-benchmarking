# LLM Production Benchmark

Industry-standard benchmark tool for evaluating LLM API deployments. Measures TTFT, TPOT, TTLT, ITL, throughput, RPS, error rate, and concurrency scaling — with a self-contained HTML report generated after every run.

---

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in this directory:

```env
API_KEY=your-api-key
API_BASE_URL=https://your-endpoint/v1
MODEL=si-qwen3-vl-30b
```

The script validates the model name at startup. Only the following models are accepted:

| Model ID | Description |
|---|---|
| `si-gpt-oss-120b` | GPT OSS 120B |
| `si-qwen3-embedding-8b` | Qwen3 Embedding 8B |
| `si-deepseek-3.2` | DeepSeek 3.2 |
| `si-qwen3.5-27b` | Qwen 3.5 27B |
| `si-qwen3-vl-30b` | Qwen3 VL 30B |
| `si-qwen3.5-35b` | Qwen 3.5 35B |
| `sains-llm-agentic` | Sains LLM Agentic |

---

## Usage

### Quick start (recommended defaults)

```bash
python production_benchmark.py
```

Runs 20 requests in `balanced` mode using the model set in `.env`.

---

### Benchmark modes

| Mode | Input tokens | Output tokens | Concurrency | Best for |
|---|---|---|---|---|
| `balanced` | 1000 | 512 | as set | General-purpose, real-world simulation |
| `ttft` | 3000 | 1 | 1 (fixed) | Measuring first-token latency |
| `throughput` | 500 | 1024 | as set | Sustained generation speed |
| `custom` | your choice | your choice | your choice | Full manual control |

---

### Common examples

**Balanced run, 30 requests, 4 concurrent workers:**
```bash
python production_benchmark.py --mode balanced --runs 30 --concurrency 4
```

**TTFT-focused measurement:**
```bash
python production_benchmark.py --mode ttft --runs 20
```

**Throughput test with high concurrency:**
```bash
python production_benchmark.py --mode throughput --concurrency 8 --runs 40
```

**Duration-based run (run for 60 seconds):**
```bash
python production_benchmark.py --mode balanced --duration 60 --concurrency 4
```

**Concurrency scaling sweep (tests c=1,2,4,8,16 automatically):**
```bash
python production_benchmark.py --mode throughput --sweep
```

**Custom sweep levels:**
```bash
python production_benchmark.py --mode throughput --sweep --sweep-levels 1,4,8,16,32 --sweep-runs 20
```

**Override a specific model:**
```bash
python production_benchmark.py --model si-deepseek-3.2 --mode balanced
```

**Save raw results to JSON as well:**
```bash
python production_benchmark.py --mode balanced --output results.json
```

---

### All options

```
--base-url        API base URL (overrides .env)
--api-key         API key (overrides .env)
--model           Model ID (must be one of the allowed models above)
--mode            balanced | ttft | throughput | custom  (default: balanced)
--concurrency     Number of parallel workers (default: 1)
--runs            Total requests to send (default: 20)
--duration        Run for N seconds instead of a fixed request count
--warmup          Warmup requests excluded from stats (default: 2)
--max-tokens      Override max output tokens
--min-tokens      Force minimum output tokens
--prompt-tokens   Override input prompt length in tokens
--output          Also save raw per-request data to a JSON file
--sweep           Run a concurrency scaling sweep
--sweep-levels    Comma-separated concurrency levels (default: 1,2,4,8,16)
--sweep-runs      Requests per sweep level (default: 20)
```

---

## Output

After each run the script saves an HTML report to the working directory:

```
YYYYMMDD_HHMMSS_<model>_benchmark.html
```

The report includes:
- KPI summary cards (TTFT P50/P95/P99, TPOT, TTLT, system throughput, RPS, error rate)
- Testing parameters table
- Latency percentile bar charts (TTFT, TPOT, TTLT)
- Per-request latency time-series
- Per-request throughput histogram
- Concurrency scaling charts (when `--sweep` is used)
- Full statistics table (mean, std, P50, P90, P95, P99, max)

The console also prints a formatted report at the end of each run.

---

## Metrics reference

| Metric | Definition |
|---|---|
| **TTFT** | Time to First Token — wall-clock time from request sent to first token received |
| **TPOT** | Time Per Output Token — average inter-token latency during generation |
| **TTLT** | Time to Last Token — total end-to-end request latency |
| **ITL** | Inter-Token Latency — per-token generation delay distribution |
| **TPS** | Tokens Per Second — generation speed per request |
| **System Throughput** | Total tokens/sec across all concurrent workers |
| **RPS** | Requests Per Second — sustained request rate |
| **Error Rate** | Percentage of requests that failed |

---

## Timeout Diagnostic Tool (Phases 1–5)

This repo includes a full industry-standard timeout diagnostic suite for LLM API deployments. It runs all five diagnostic phases (connectivity, TTFT, stall, E2E, concurrency) and generates a detailed HTML report for each model.

### Runner script

**To run all phases for all models:**

```powershell
# Windows PowerShell
.\run_timeout_test.ps1
```

**To run for a single model:**

```powershell
.\run_timeout_test.ps1 -Model "Qwen/Qwen3-VL-30B-A3B-Instruct"
```

**To run only the smoke test (TC-03 baseline):**

```powershell
.\run_timeout_test.ps1 -Single
```

**To override SLO thresholds:**

```powershell
.\run_timeout_test.ps1 -SloTtft 2000 -SloTpot 100 -StallMs 3000
```

### Output

Each run saves a full HTML diagnostic report to:

```
results/timeout_test/<timestamp>_<model>_timeout_diag.html
```

The report includes:
- KPI summary (TTFT, TPOT, connect, stall, timeout counts)
- Root-cause analysis and recommendations
- SLO compliance bars
- Per-test latency scatter
- Max chunk gap (stall detector)
- ITL histogram
- Concurrency sweep charts
- Full results table

See [timeout_test.py](timeout_test.py) for full methodology and test details.
