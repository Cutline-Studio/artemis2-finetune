#!/bin/bash
set -eo pipefail
cd /root/artemis2-finetune
export PYTHONPATH=scripts HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
t0=$(date +%s)
nvidia-smi
python -m pip install --break-system-packages --no-cache-dir -q \
  transformers==5.20.2 trl==1.16.0 peft==0.21.2 datasets==5.1.0 accelerate==1.15.0 \
  bitsandbytes==0.50.2 safetensors==0.8.0 huggingface_hub==2.3.0 tokenizers==0.23.3 \
  pyyaml==6.0.3 sentencepiece protobuf
python -m pip install --break-system-packages --no-cache-dir -q flash-linear-attention || echo "fla install failed (fallback kernels)"
python -c "import torch,transformers,trl,peft;print('versions',torch.__version__,transformers.__version__,trl.__version__,peft.__version__)"
echo "setup_seconds=$(( $(date +%s) - t0 ))"
start=$(date +%s)
python scripts/train_sft.py --config configs/a6000_small.yaml
echo "train_seconds=$(( $(date +%s) - start ))"
rm -rf outputs/a6000-run/checkpoint-*
e0=$(date +%s)
python scripts/eval_prompts.py --base Qwen/Qwen3.5-4B --adapter outputs/a6000-run/adapter --max-samples 37 --out outputs/a6000-run/eval_report.json
echo "eval_seconds=$(( $(date +%s) - e0 ))"
echo "total_seconds=$(( $(date +%s) - t0 ))"
echo RUN_COMPLETE
