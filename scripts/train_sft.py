"""LoRA/QLoRA SFT for Artemis 2.0 (Qwen3.8-27B on H100, or Qwen3.5-0.8B for CPU smoke).

Usage:
  python scripts/train_sft.py --config configs/train_qlora_h100.yaml
  python scripts/train_sft.py --config configs/train_smoke_cpu.yaml
"""
from __future__ import annotations

import argparse, json, os, random, sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_and_tokenizer(cfg: dict):
    model_id = cfg["model_name_or_path"]
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = None
    if cfg.get("load_in_4bit"):
        compute = getattr(torch, cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
        )

    dtype = torch.bfloat16 if cfg.get("bf16") else (torch.float16 if cfg.get("fp16") else torch.float32)
    device_map = "auto" if (torch.cuda.is_available() or quant is not None) else None

    # Qwen3.5/3.8 ship as Qwen3_5ForConditionalGeneration (VLM). For text SFT,
    # AutoModelForCausalLM resolves to Qwen3_5ForCausalLM on recent transformers.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=quant,
        # With 4-bit, still pin the dtype of non-quantized modules (fp16 on T4, which has no bf16).
        dtype=dtype if (quant is None or cfg.get("bf16") or cfg.get("fp16")) else None,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(cfg["lora_target_modules"]),
    )
    model = get_peft_model(model, lora)
    # fp16 AMP (T4) needs fp32 master weights for trainable params: GradScaler cannot
    # unscale bf16/fp16 grads (Azure ML run crimson_shirt_tfhzb9hm6s failed with
    # "_amp_foreach_non_finite_check_and_unscale_cuda not implemented for 'BFloat16'").
    if cfg.get("fp16"):
        for p in model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
    from collections import Counter
    print(json.dumps({"event": "trainable_dtypes", "counts": dict(Counter(str(p.dtype) for p in model.parameters() if p.requires_grad))}), flush=True)
    model.print_trainable_parameters()
    return model, tok


def load_splits(cfg: dict, tokenizer=None):
    """Load conversational JSONL. Keep `messages` so TRL can apply assistant-only loss."""
    data_files = {"train": str(ROOT / cfg["train_file"])}
    if cfg.get("eval_file") and (ROOT / cfg["eval_file"]).exists():
        data_files["eval"] = str(ROOT / cfg["eval_file"])
    ds = load_dataset("json", data_files=data_files)
    # Drop non-message columns; TRL conversational path only needs messages.
    def keep_messages(batch):
        return {"messages": batch["messages"]}
    for split in list(ds.keys()):
        ds[split] = ds[split].map(keep_messages, batched=True, remove_columns=[c for c in ds[split].column_names if c != "messages"])
    if cfg.get("max_train_samples"):
        n = min(int(cfg["max_train_samples"]), len(ds["train"]))
        ds["train"] = ds["train"].shuffle(seed=int(cfg.get("seed", 0))).select(range(n))
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(int(cfg.get("seed", 0)))
    out = Path(args.output_dir or cfg["output_dir"])
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    (out / "used_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    print(json.dumps({
        "event": "start",
        "model": cfg["model_name_or_path"],
        "cuda": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "output": str(out),
    }), flush=True)

    model, tok = build_model_and_tokenizer(cfg)
    ds = load_splits(cfg, tok)

    sft_args = SFTConfig(
        output_dir=str(out),
        run_name=cfg.get("run_name"),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=int(cfg["warmup_steps"]) if cfg.get("warmup_steps") is not None else 0,
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        max_steps=int(cfg["max_steps"]) if cfg.get("max_steps") is not None else -1,
        logging_steps=int(cfg.get("logging_steps", 10)),
        eval_strategy=cfg.get("eval_strategy", "no"),
        eval_steps=int(cfg["eval_steps"]) if cfg.get("eval_steps") else None,
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=int(cfg.get("save_steps", 100)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        bf16=bool(cfg.get("bf16", False)),
        fp16=bool(cfg.get("fp16", False)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
        max_length=int(cfg.get("max_seq_length", 2048)),
        packing=bool(cfg.get("packing", False)),
        report_to=[],
        seed=int(cfg.get("seed", 0)),
        completion_only_loss=bool(cfg.get("completion_only_loss", True)),
        assistant_only_loss=bool(cfg.get("assistant_only_loss", False)),
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 0)),
        chat_template_path=str(ROOT / cfg["chat_template_path"]) if cfg.get("chat_template_path") else None,
    )

    eval_ds = ds["eval"] if "eval" in ds and cfg.get("eval_strategy", "no") != "no" else None
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds["train"],
        eval_dataset=eval_ds,
        processing_class=tok,
    )
    train_result = trainer.train()
    trainer.save_model(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))
    metrics = dict(train_result.metrics)
    (out / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"event": "done", "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()
