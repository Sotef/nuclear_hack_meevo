#!/usr/bin/env python
"""LoRA SFT trainer for MiniCPM5-2B (and compatible LlamaForCausalLM chat models).

Trains on a JSONL file where each line is {"messages": [{"role","content"}, ...]}
using the model's own chat template with enable_thinking=false. Loss is computed
on assistant spans only (fallback: full sequence if no assistant span is found).

Deterministic: seed is applied to python/numpy/torch/cuda; versions are logged.
Progress (HF Trainer + tqdm) is written to --log.

Example:
  .venv\\Scripts\\python.exe tools\\train_llm\\train_llm.py ^
    --model models\\llm_minicpm5\\base --data data\\twix\\train_smoke.jsonl ^
    --out models\\llm_minicpm5\\smoke --epochs 1 --lr 2e-4 --lora-r 16 ^
    --lora-alpha 32 --max-seq-len 512 --batch 1 --grad-accum 4 ^
    --log logs\\train_minicpm5_smoke.log --force
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import sys
import time
from pathlib import Path

TOOL_VERSION = "1.0.0"
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
ASSISTANT_START_RE = re.compile(r"<\|im_start\|>assistant\n")
ASSISTANT_END = "<|im_end|>"


def die(msg: str, code: int = 1) -> "NoReturn":
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="LoRA SFT trainer for MiniCPM5-2B on T-Wix-style messages JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, help="HF model id or local directory")
    p.add_argument("--data", required=True, help="JSONL with {'messages': [...]} per line")
    p.add_argument("--out", required=True, help="output directory for LoRA adapter")
    p.add_argument("--epochs", type=float, default=1.0, help="number of epochs")
    p.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--max-seq-len", type=int, default=512, help="max sequence length")
    p.add_argument("--batch", type=int, default=1, help="per-device train batch size")
    p.add_argument("--grad-accum", type=int, default=4, help="gradient accumulation steps")
    p.add_argument("--log", required=True, help="log file (Trainer/tqdm output is teed here)")
    p.add_argument("--resume", action="store_true", help="resume from latest checkpoint in --out")
    p.add_argument("--force", action="store_true", help="overwrite non-empty --out")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--save-steps", type=int, default=500, help="checkpoint interval (steps)")
    p.add_argument("--max-steps", type=int, default=-1, help="hard cap on optimizer steps (-1 = no cap)")
    p.add_argument("--no-grad-checkpointing", action="store_true", help="disable gradient checkpointing")
    p.add_argument("--attn-implementation", default="sdpa", help="attention implementation")
    p.add_argument("--version", action="version", version=f"train_llm {TOOL_VERSION}")
    return p.parse_args(argv)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_versions(log_fh) -> dict:
    import torch
    import transformers
    import peft
    try:
        import accelerate
        acc = accelerate.__version__
    except Exception:
        acc = "not installed"
    info = {
        "tool": f"train_llm {TOOL_VERSION}",
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "accelerate": acc,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 3)
        if torch.cuda.is_available() else None,
    }
    for k, v in info.items():
        print(f"[env] {k}: {v}", file=log_fh)
    return info


class JsonlMessagesDataset:
    """Lazy JSONL dataset with a byte-offset index (memory-safe for 3.7 GB files)."""

    def __init__(self, path: Path, tokenizer, max_seq_len: int):
        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.offsets: list[int] = []
        with path.open("rb") as fh:
            pos = 0
            for line in fh:
                if line.strip():
                    self.offsets.append(pos)
                pos += len(line)
        self._fh = path.open("rb")
        self.skipped = 0

    def __len__(self) -> int:
        return len(self.offsets)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def _read(self, idx: int) -> dict:
        self._fh.seek(self.offsets[idx])
        line = self._fh.readline()
        return json.loads(line)

    def _encode(self, rec: dict) -> dict:
        msgs = rec.get("messages") or []
        clean = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", ""))
            content = m.get("content", "")
            if not isinstance(content, str):
                content = "" if content is None else str(content)
            clean.append({"role": role, "content": content})
        if not clean:
            clean = [{"role": "user", "content": ""}]
        try:
            text = self.tokenizer.apply_chat_template(
                clean, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                clean, tokenize=False, add_generation_prompt=False
            )
        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_len,
            return_offsets_mapping=True,
        )
        ids = list(enc["input_ids"])
        offsets = list(enc["offset_mapping"])
        if not ids:
            ids = [self.tokenizer.bos_token_id or 0]
            offsets = [(0, 0)]
        labels = [-100] * len(ids)

        regions = []
        for m in ASSISTANT_START_RE.finditer(text):
            end = text.find(ASSISTANT_END, m.end())
            if end != -1:
                regions.append((m.end(), end + len(ASSISTANT_END)))
        if regions:
            for i, (s, e) in enumerate(offsets):
                if e <= s:
                    continue
                for rs, re_ in regions:
                    if e > rs and s < re_:
                        labels[i] = ids[i]
                        break
        if all(lab == -100 for lab in labels):
            labels = list(ids)
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": labels,
        }

    def __getitem__(self, idx: int) -> dict:
        try:
            rec = self._read(idx)
            return self._encode(rec)
        except Exception as exc:
            self.skipped += 1
            print(f"[data] skip line {idx}: {exc}", file=sys.stderr)
            tok = self.tokenizer.bos_token_id or 0
            return {"input_ids": [tok], "attention_mask": [1], "labels": [tok]}


class PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, attn, labels = [], [], []
        for f in features:
            pad = maxlen - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def find_latest_checkpoint(out_dir: Path) -> Path | None:
    ckpts = []
    for p in out_dir.glob("checkpoint-*"):
        if p.is_dir():
            try:
                ckpts.append((int(p.name.split("-")[-1]), p))
            except ValueError:
                continue
    if not ckpts:
        return None
    return max(ckpts, key=lambda x: x[0])[1]


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model, PeftModel

    data_path = Path(args.data)
    out_dir = Path(args.out)
    log_path = Path(args.log)
    if not data_path.is_file():
        die(f"data file not found: {data_path}")
    if not torch.cuda.is_available():
        die("CUDA is not available in this interpreter; training requires a CUDA GPU")
    if args.max_seq_len < 64:
        die("--max-seq-len must be >= 64")
    if args.batch < 1 or args.grad_accum < 1:
        die("--batch and --grad-accum must be >= 1")

    if out_dir.exists() and any(out_dir.iterdir()):
        if args.resume:
            pass
        elif args.force:
            print(f"[out] --force: existing contents of {out_dir} will be overwritten", file=sys.stderr)
        else:
            die(f"output directory is not empty: {out_dir} (use --force to overwrite or --resume)")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_mode = "a" if args.resume else "w"
    log_fh = log_path.open(log_mode, encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = Tee(old_out, log_fh)
    sys.stderr = Tee(old_err, log_fh)
    try:
        print(f"\n===== train_llm {TOOL_VERSION} start {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
        print(f"[args] {vars(args)}")
        env = log_versions(log_fh)
        set_seed(args.seed)
        print(f"[seed] {args.seed}")

        t0 = time.time()
        print(f"[model] loading {args.model} (bf16, attn={args.attn_implementation})")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation=args.attn_implementation,
        )
        model.config.use_cache = False
        print(f"[model] loaded in {time.time()-t0:.1f}s; params={sum(p.numel() for p in model.parameters())/1e9:.3f}B")

        resume_ckpt = None
        if args.resume:
            resume_ckpt = find_latest_checkpoint(out_dir)
            if resume_ckpt is None:
                die(f"--resume requested but no checkpoint-* found in {out_dir}")
            print(f"[resume] latest checkpoint: {resume_ckpt}")

        if args.resume and (out_dir / "adapter_config.json").is_file():
            model = PeftModel.from_pretrained(model, str(out_dir), is_trainable=True)
            print("[lora] loaded adapter from out dir")
        else:
            lora_cfg = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=DEFAULT_TARGET_MODULES,
            )
            model = get_peft_model(model, lora_cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[lora] trainable={trainable:,} ({100*trainable/total:.3f}% of {total:,})")

        if not args.no_grad_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            print("[mem] gradient checkpointing enabled")

        print(f"[data] indexing {data_path}")
        dataset = JsonlMessagesDataset(data_path, tokenizer, args.max_seq_len)
        print(f"[data] {len(dataset):,} records indexed")

        import math
        steps_per_epoch = math.ceil(len(dataset) / (args.batch * args.grad_accum))
        total_steps = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
        warmup_steps = max(1, int(0.03 * total_steps))
        print(f"[train] steps/epoch={steps_per_epoch} total_steps={total_steps} warmup_steps={warmup_steps}")

        training_args = TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            weight_decay=0.0,
            max_grad_norm=1.0,
            bf16=True,
            fp16=False,
            logging_steps=10,
            logging_first_step=True,
            save_strategy="steps" if args.save_steps > 0 else "no",
            save_steps=max(1, args.save_steps),
            save_total_limit=1,
            save_only_model=True,
            report_to=[],
            remove_unused_columns=False,
            dataloader_num_workers=0,
            seed=args.seed,
            data_seed=args.seed,
            gradient_checkpointing=False,
            optim="adamw_torch",
            disable_tqdm=False,
            log_level="info",
            run_name=out_dir.name,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=PadCollator(tokenizer.pad_token_id),
            processing_class=tokenizer,
        )

        print(f"[train] start {time.strftime('%Y-%m-%d %H:%M:%S')}")
        t_train = time.time()
        train_result = trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)
        train_s = time.time() - t_train
        print(f"[train] done in {train_s:.1f}s ({train_s/3600:.2f}h)")
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[mem] peak_cuda_gb={peak_gb:.3f}")

        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        print(f"[save] adapter + tokenizer -> {out_dir}")

        summary = {
            "tool": f"train_llm {TOOL_VERSION}",
            "model": args.model,
            "data": str(data_path),
            "out": str(out_dir),
            "records": len(dataset),
            "skipped": dataset.skipped,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "max_seq_len": args.max_seq_len,
            "batch": args.batch,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch * args.grad_accum,
            "seed": args.seed,
            "train_seconds": round(train_s, 1),
            "train_hours": round(train_s / 3600, 3),
            "peak_cuda_gb": round(peak_gb, 3),
            "train_loss": train_result.training_loss,
            "global_step": train_result.global_step,
            "log_history": trainer.state.log_history,
            "env": env,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (out_dir / "train_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[summary] train_loss={train_result.training_loss} global_step={train_result.global_step} peak_cuda_gb={peak_gb:.3f}")
        print(f"===== train_llm done {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
        dataset.close()
        return 0
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        log_fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
