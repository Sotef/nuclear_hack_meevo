#!/usr/bin/env python3
"""export_llm — merge a PEFT LoRA adapter into a Llama-architecture base model and
export it to an on-device LLM runtime.

Supported formats:
  ort-genai-int4  ONNX Runtime GenAI model builder, int4 (block 32, accuracy 4)
  litert-int4     LiteRT-LM export via `litert-torch export_hf` (if installed)

Build-time only: uses torch / transformers / peft / onnxruntime-genai from the
project .venv. The produced artifact is what ships to Android.

Examples
--------
  # merge LoRA only (bf16 safetensors)
  python tools/export_llm/export_llm.py --base C:\\temp\\opencode\\minicpm5_base \
      --lora models\\llm_minicpm5\\smoke --out models\\llm_minicpm5\\smoke_merged \
      --format merge-only --force

  # merge + ORT GenAI int4 + parity on 10 examples
  python tools/export_llm/export_llm.py --base C:\\temp\\opencode\\minicpm5_base \
      --lora models\\llm_minicpm5\\smoke --out models\\llm_minicpm5\\smoke_int4 \
      --merged-out models\\llm_minicpm5\\smoke_merged --format ort-genai-int4 \
      --verify 10 --force
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = ROOT / "prompts" / "postproc_v1.md"
DEFAULT_EVAL = ROOT / "data" / "postproc" / "punct_eval.jsonl"


def log(msg: str) -> None:
    print(f"[export_llm] {msg}", flush=True)


def die(msg: str, code: int = 1) -> "NoReturn":
    print(f"[export_llm] ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def check_input_dir(path: Path, what: str) -> Path:
    if not path.is_dir():
        die(f"{what} directory not found: {path}")
    return path


def prepare_out(path: Path, force: bool) -> Path:
    if path.exists() and any(path.iterdir()):
        if not force:
            die(f"output directory is not empty: {path} (use --force to overwrite)", 2)
        log(f"--force: removing existing {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def versions() -> dict:
    import torch
    import transformers
    import peft
    import onnxruntime_genai

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "onnxruntime_genai": onnxruntime_genai.__version__,
    }


def merge_lora(base: Path, lora: Path, merged_out: Path, force: bool) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    prepare_out(merged_out, force)
    log(f"loading tokenizer from {base}")
    tokenizer = AutoTokenizer.from_pretrained(str(base), trust_remote_code=False)
    log(f"loading base model (bf16, cpu) from {base}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(base),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    log(f"base loaded in {time.time() - t0:.1f}s")
    log(f"loading LoRA adapter from {lora}")
    model = PeftModel.from_pretrained(model, str(lora))
    log("merging adapter (merge_and_unload)")
    t0 = time.time()
    model = model.merge_and_unload()
    log(f"merged in {time.time() - t0:.1f}s")
    log(f"saving merged model to {merged_out}")
    t0 = time.time()
    model.save_pretrained(str(merged_out), safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(str(merged_out))
    # Preserve the model's chat template (transformers may not copy .jinja).
    for name in ("chat_template.jinja", "generation_config.json", "special_tokens_map.json"):
        src = base / name
        if src.is_file() and not (merged_out / name).exists():
            shutil.copy2(src, merged_out / name)
    log(f"saved in {time.time() - t0:.1f}s")
    return merged_out


def export_ort_genai_int4(merged: Path, out: Path, force: bool, cache_dir: Path) -> Path:
    from onnxruntime_genai.models import builder

    prepare_out(out, force)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log(f"ORT GenAI builder: llama int4 (block 32, accuracy 4) -> {out}")
    t0 = time.time()
    opts = builder.parse_extra_options(
        model_name="llama",
        input_path=str(merged),
        output_dir=str(out),
        precision="int4",
        execution_provider="cpu",
        cache_dir=str(cache_dir),
        extra_options=["block_size=32", "accuracy_level=4", "embedding_quant_type=int4"],
    )
    builder.create_model(
        model_name="llama",
        input_path=str(merged),
        output_dir=str(out),
        precision="int4",
        execution_provider="cpu",
        cache_dir=str(cache_dir),
        **opts,
    )
    log(f"ORT GenAI export done in {time.time() - t0:.1f}s")
    return out


def export_litert_int4(merged: Path, out: Path, force: bool) -> Path:
    import subprocess

    prepare_out(out, force)
    exe = shutil.which("litert-torch")
    if not exe:
        die(
            "litert-int4 requested but `litert-torch` is not installed. "
            "Install google-ai-edge/LiteRT-LM (pip install litert-torch) or use "
            "--format ort-genai-int4."
        )
    cmd = [exe, "export_hf", str(merged), "--output_dir", str(out),
           "--quantization_recipe", "dynamic_wi8_afp32"]
    log("running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out


def parse_prompt(path: Path, mode: str = "punctuate") -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    marker = f"## {mode}"
    if marker not in text:
        die(f"mode '{mode}' not found in prompt file {path}")
    block = text.split(marker, 1)[1]
    if "\n## " in block:
        block = block.split("\n## ", 1)[0]
    system = ""
    user = ""
    for line in block.splitlines():
        if line.startswith("SYSTEM:"):
            system = line[len("SYSTEM:"):].strip()
        elif line.startswith("USER:"):
            user = line[len("USER:"):].strip()
    if not system or not user:
        die(f"could not parse SYSTEM/USER for mode '{mode}' in {path}")
    return system, user


def build_prompt(tokenizer, system: str, user: str, text: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user.replace("{text}", text)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def load_eval(path: Path, n: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    if not rows:
        die(f"no examples read from {path}")
    return rows


def run_torch(model_dir: Path, prompts: list[str], max_new: int, device: str) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"parity: loading torch model from {model_dir} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()
    outs = []
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[1, 130073],
            )
        text = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outs.append(text.strip())
        log(f"  torch {i + 1}/{len(prompts)}: {len(text)} chars")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return outs


def run_ort_genai(model_dir: Path, prompts: list[str], max_new: int) -> list[str]:
    import onnxruntime_genai as og

    log(f"parity: loading ORT GenAI model from {model_dir}")
    model = og.Model(str(model_dir))
    tokenizer = og.Tokenizer(model)
    outs = []
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt)
        params = og.GeneratorParams(model)
        params.set_search_options(
            do_sample=False,
            max_length=len(input_ids) + max_new,
            min_length=len(input_ids),
        )
        generator = og.Generator(model, params)
        generator.append_tokens(input_ids)
        new_tokens = []
        while not generator.is_done():
            generator.generate_next_token()
            tok = generator.get_next_tokens()[0]
            new_tokens.append(tok)
        text = tokenizer.decode(new_tokens)
        outs.append(text.strip())
        log(f"  ort {i + 1}/{len(prompts)}: {len(text)} chars")
    return outs


def verify_parity(
    merged: Path,
    int4: Path,
    eval_path: Path,
    prompt_path: Path,
    n: int,
    max_new: int,
    device: str,
    report_path: Path,
) -> dict:
    system, user = parse_prompt(prompt_path, "punctuate")
    rows = load_eval(eval_path, n)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(merged), trust_remote_code=False)
    prompts = [build_prompt(tok, system, user, r.get("text_raw") or r.get("text") or "") for r in rows]
    torch_out = run_torch(merged, prompts, max_new, device)
    ort_out = run_ort_genai(int4, prompts, max_new)
    sims = [difflib.SequenceMatcher(None, a, b).ratio() for a, b in zip(torch_out, ort_out)]
    exact = sum(1 for a, b in zip(torch_out, ort_out) if a == b)
    report = {
        "n": len(rows),
        "exact_matches": exact,
        "mean_char_similarity": sum(sims) / len(sims) if sims else 0.0,
        "per_example": [
            {
                "id": r.get("id"),
                "char_similarity": s,
                "exact": a == b,
                "torch": a,
                "ort_int4": b,
            }
            for r, a, b, s in zip(rows, torch_out, ort_out, sims)
        ],
        "prompt_mode": "punctuate",
        "max_new_tokens": max_new,
        "torch_device": device,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"parity: exact {exact}/{len(rows)}, mean char-similarity {report['mean_char_similarity']:.4f}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Merge a PEFT LoRA adapter and export a Llama-architecture model "
                    "to an on-device runtime (ORT GenAI int4 / LiteRT int4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--base", required=True, help="base model directory (HF format)")
    ap.add_argument("--lora", required=True, help="PEFT LoRA adapter directory")
    ap.add_argument("--out", required=True, help="output directory for the exported model")
    ap.add_argument(
        "--format",
        default="ort-genai-int4",
        choices=["ort-genai-int4", "litert-int4", "merge-only"],
        help="target format (default: ort-genai-int4)",
    )
    ap.add_argument("--merged-out", default=None,
                    help="optional directory to save the merged bf16 model")
    ap.add_argument("--merged-in", default=None,
                    help="use an already merged model directory and skip the merge step")
    ap.add_argument("--verify", type=int, default=0, metavar="N",
                    help="run parity check on N examples from --eval-file (0 = off)")
    ap.add_argument("--eval-file", default=str(DEFAULT_EVAL),
                    help="JSONL with text_raw/text for parity (default: data/postproc/punct_eval.jsonl)")
    ap.add_argument("--prompt-file", default=str(DEFAULT_PROMPT),
                    help="prompt templates file (default: prompts/postproc_v1.md)")
    ap.add_argument("--max-new-tokens", type=int, default=128,
                    help="max new tokens for parity generation (default: 128)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="torch device for parity (default: cuda, falls back to cpu)")
    ap.add_argument("--cache-dir", default=str(ROOT / "models" / "llm_build_cache"),
                    help="builder cache directory")
    ap.add_argument("--force", action="store_true", help="overwrite non-empty output directories")
    args = ap.parse_args()

    base = check_input_dir(Path(args.base), "base")
    lora = check_input_dir(Path(args.lora), "lora")
    out = Path(args.out)
    merged_out = Path(args.merged_out) if args.merged_out else None

    log("versions: " + json.dumps(versions(), ensure_ascii=False))
    log(f"base={base} lora={lora} out={out} format={args.format}")

    if args.merged_in:
        merged_out = check_input_dir(Path(args.merged_in), "merged-in")
        log(f"using existing merged model: {merged_out} (merge step skipped)")
    elif args.format == "merge-only":
        if merged_out is None:
            merged_out = out
        merge_lora(base, lora, merged_out, args.force)
        log("merge-only done")
        return 0
    else:
        if merged_out is None:
            merged_out = out.parent / (out.name + "_merged")
        merge_lora(base, lora, merged_out, args.force)

    if args.format == "ort-genai-int4":
        export_ort_genai_int4(merged_out, out, args.force, Path(args.cache_dir))
    elif args.format == "litert-int4":
        export_litert_int4(merged_out, out, args.force)
    else:
        die(f"unknown format {args.format}")

    if args.verify > 0:
        device = args.device
        try:
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                log("CUDA not available, parity falls back to CPU")
                device = "cpu"
        except Exception:
            device = "cpu"
        report_path = out / "parity_report.json"
        verify_parity(
            merged_out, out, Path(args.eval_file), Path(args.prompt_file),
            args.verify, args.max_new_tokens, device, report_path,
        )
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
