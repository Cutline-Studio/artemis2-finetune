.PHONY: data smoke train-h100 eval-h100 merge-h100

data:
	PYTHONPATH=scripts python scripts/merge_generated.py
	PYTHONPATH=scripts python scripts/build_dataset.py --generated data/raw/generated_all.jsonl
	PYTHONPATH=scripts python scripts/write_data_card.py

smoke:
	PYTHONPATH=scripts python scripts/train_sft.py --config configs/train_smoke_cpu.yaml

train-h100:
	PYTHONPATH=scripts python scripts/train_sft.py --config configs/train_qlora_h100.yaml

eval-h100:
	PYTHONPATH=scripts python scripts/eval_prompts.py --base Qwen/Qwen3.8-27B --adapter outputs/artemis2-qlora/adapter --load-4bit

merge-h100:
	PYTHONPATH=scripts python scripts/merge_lora.py --base Qwen/Qwen3.8-27B --adapter outputs/artemis2-qlora/adapter --out outputs/artemis2-merged-bf16
