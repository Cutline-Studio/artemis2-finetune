# CPU smoke test report

Pipeline proof on the shared box (no GPU, no cloud spend).

## Command

```bash
cd /workspace/artemis2-finetune
source .venv/bin/activate
export HF_HOME=/workspace/.hf-cache-artemis2
PYTHONPATH=scripts python scripts/build_dataset.py
PYTHONPATH=scripts python -u scripts/train_sft.py --config configs/train_smoke_cpu.yaml
PYTHONPATH=scripts python -u  # eval probes (see smoke/raw_eval.log)
```

## Versions

- python: 3.12.15
- platform: Linux-6.12.94+-x86_64-with-glibc2.41
- torch: 2.14.1+cpu (cuda=False)
- transformers: 5.20.2
- trl: 1.16.0
- peft: 0.21.2
- datasets: 5.1.0
- accelerate: 1.15.0

## Config

- model: `Qwen/Qwen3.5-0.8B` (same `qwen3_5` family as Qwen3.8-27B)
- seed: `20261025`
- max_steps: 6
- max_train_samples: 24
- max_seq_length: 1536
- LoRA r=8, target modules include hybrid attention + MLP

## Train metrics (verbatim from train_metrics.json)

```json
{
  "train_runtime": 47.3529,
  "train_samples_per_second": 0.127,
  "train_steps_per_second": 0.127,
  "total_flos": 10936555875840.0,
  "train_loss": 0.9077735940615336,
  "epoch": 0.25
}
```

## Step losses (from raw_train.log)

```
{'loss': '0.7116', 'grad_norm': '8.988', 'learning_rate': '0.0002', 'entropy': '0.4277', 'mean_token_accuracy': '0.825', 'num_tokens': '622', 'epoch': '0.04167'}
{'loss': '1.311', 'grad_norm': '8.566', 'learning_rate': '0.0001866', 'entropy': '1.344', 'mean_token_accuracy': '0.56', 'num_tokens': '1221', 'epoch': '0.08333'}
{'loss': '0.3205', 'grad_norm': '5.42', 'learning_rate': '0.00015', 'entropy': '0.6371', 'mean_token_accuracy': '0.8571', 'num_tokens': '1831', 'epoch': '0.125'}
{'loss': '0.3542', 'grad_norm': '6.969', 'learning_rate': '0.0001', 'entropy': '0.7434', 'mean_token_accuracy': '0.9048', 'num_tokens': '2426', 'epoch': '0.1667'}
{'loss': '2.386', 'grad_norm': '14.32', 'learning_rate': '5e-05', 'entropy': '2.208', 'mean_token_accuracy': '0.45', 'num_tokens': '3025', 'epoch': '0.2083'}
{'loss': '0.3636', 'grad_norm': '4.785', 'learning_rate': '1.34e-05', 'entropy': '0.6016', 'mean_token_accuracy': '0.8889', 'num_tokens': '3620', 'epoch': '0.25'}
{"event": "done", "metrics": {"train_runtime": 47.3529, "train_samples_per_second": 0.127, "train_steps_per_second": 0.127, "total_flos": 10936555875840.0, "train_loss": 0.9077735940615336, "epoch": 0.25}}
```

## Eval probes (adapter loaded)

```json
{
  "probes": [
    {
      "q": "Who are you?",
      "a": "I am Artemis, a proprietary AI assistant built by Cutline Industries. I merge full-stack code generation, context-aware memory, and multi-model routing into one platform.\n\nI was created by Latoya Pittman, software engineer and CEO of Cutline Industries. I am an open-weight foundation model that Cutline Industries fine-tuned to be Artemis. It is not trained from scratch.",
      "latency_s": 6.32
    },
    {
      "q": "Write a Python function that returns the factorial of n.",
      "a": "```python\ndef factorial(n):\n    if n < 0:\n        raise ValueError(\"Factorial is not defined for negative numbers.\")\n    if n == 0 or n == 1:\n        return 1\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n```",
      "latency_s": 6.36
    }
  ]
}

```

## Outcome

Data loading, chat-template assistant-only loss, LoRA training, adapter save, and generate-with-adapter all succeeded on CPU.
