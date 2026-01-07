python production_benchmark.py --mode ttft --model "deepseek-ai/DeepSeek-V3.2" --max-tokens 512 --runs 15 --output ttft_deepseek_benchmark.json
python production_benchmark.py --mode ttft --model "Qwen/Qwen3-VL-30B-A3B-Instruct" --max-tokens 512 --runs 15 --output ttft_qwen_benchmark.json
python production_benchmark.py --mode ttft --model "openai/gpt-oss-120b" --max-tokens 512 --runs 15 --output ttft_openai_benchmark.json

python production_benchmark.py --mode throughput --model "deepseek-ai/DeepSeek-V3.2" --concurrency 5 --duration 900 --max-tokens 512 --min-tokens 256 --output c5_deepseek_benchmark.json
python production_benchmark.py --mode throughput --model "Qwen/Qwen3-VL-30B-A3B-Instruct" --concurrency 5 --duration 900 --max-tokens 512 --min-tokens 256 --output c5_qwen_benchmark.json
python production_benchmark.py --mode throughput --model "openai/gpt-oss-120b" --concurrency 5 --duration 900 --max-tokens 512 --min-tokens 256 --output c5_openai_benchmark.json

python production_benchmark.py --mode throughput --model "deepseek-ai/DeepSeek-V3.2" --concurrency 10 --duration 900 --max-tokens 512 --min-tokens 256 --output c10_deepseek_benchmark.json
python production_benchmark.py --mode throughput --model "Qwen/Qwen3-VL-30B-A3B-Instruct" --concurrency 10 --duration 900 --max-tokens 512 --min-tokens 256 --output c10_qwen_benchmark.json
python production_benchmark.py --mode throughput --model "openai/gpt-oss-120b" --concurrency 10 --duration 900 --max-tokens 512 --min-tokens 256 --output c10_openai_benchmark.json

python production_benchmark.py --mode throughput --model "deepseek-ai/DeepSeek-V3.2" --concurrency 15 --duration 900 --max-tokens 512 --min-tokens 256 --output c15_deepseek_benchmark.json
python production_benchmark.py --mode throughput --model "Qwen/Qwen3-VL-30B-A3B-Instruct" --concurrency 15 --duration 900 --max-tokens 512 --min-tokens 256 --output c15_qwen_benchmark.json
python production_benchmark.py --mode throughput --model "openai/gpt-oss-120b" --concurrency 15 --duration 900 --max-tokens 512 --min-tokens 256 --output c15_openai_benchmark.json
