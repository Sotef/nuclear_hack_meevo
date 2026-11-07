#!/usr/bin/env python
"""Deterministic LLM post-processor for Russian ASR transcripts.

Backends:
  torch  transformers + PyTorch (CUDA bf16/fp16 or CPU fp32), manual greedy loop
  onnx   ONNX Runtime GenAI package directory (genai_config.json)

Modes:
  punctuate  restore punctuation/capitalization only
  correct    fix ASR errors + punctuation
  full       correction + doubtful flags + edits (JSON output)

Generation is greedy (do_sample=False, temperature=0) and deterministic.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

TOOL_VERSION = "1.1.0"
DEFAULT_TEMPLATES = Path(__file__).resolve().parents[2] / "prompts" / "postproc_v1.md"
THINK_RE = re.compile(r"^\s*<think\b[^>]*>.*?</think\s*>\s*", re.DOTALL | re.IGNORECASE)
INLINE_DOUBTFUL_RE = re.compile(r"\[doubtful:\s*([^\]]+)\]", re.IGNORECASE)


def die(msg: str, code: int = 1) -> "NoReturn":
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def parse_templates(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        die(f"prompt templates not found: {path}")
    text = path.read_text(encoding="utf-8-sig")
    sections: dict[str, dict[str, str]] = {}
    current = None
    field = None
    buf: list[str] = []

    def flush_field() -> None:
        nonlocal buf, field
        if current is not None and field is not None:
            sections[current][field] = "\n".join(buf).strip()
        buf = []
        field = None

    for raw in text.splitlines():
        line = raw.rstrip()
        m = re.match(r"^##\s+([A-Za-z0-9_]+)\s*$", line)
        if m:
            flush_field()
            current = m.group(1).lower()
            sections[current] = {}
            field = None
            continue
        if current is None:
            continue
        if line.strip().upper() == "SYSTEM:":
            flush_field()
            field = "system"
            continue
        if line.strip().upper() == "USER:":
            flush_field()
            field = "user"
            continue
        if field is not None:
            buf.append(line)
    flush_field()

    for mode in ("punctuate", "correct", "full"):
        if mode not in sections or "system" not in sections[mode] or "user" not in sections[mode]:
            die(f"template file {path} has no complete '{mode}' section (SYSTEM/USER)")
        if "{text}" not in sections[mode]["user"]:
            die(f"template '{mode}' USER section must contain {{text}}")
    return sections


def read_items(in_path: Path) -> list[dict]:
    if not in_path.exists():
        die(f"input not found: {in_path}")
    items: list[dict] = []
    if in_path.is_dir():
        files = sorted(in_path.glob("*.txt"))
        if not files:
            die(f"no *.txt files in directory: {in_path}")
        for f in files:
            items.append({"id": f.stem, "text": f.read_text(encoding="utf-8-sig").strip()})
        return items
    if in_path.suffix.lower() == ".jsonl":
        with in_path.open(encoding="utf-8-sig") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    die(f"{in_path}:{lineno}: invalid JSON: {exc}")
                text = obj.get("text") or obj.get("text_raw") or obj.get("text_asr")
                if not isinstance(text, str) or not text.strip():
                    die(f"{in_path}:{lineno}: missing non-empty 'text'/'text_raw'/'text_asr'")
                items.append({"id": str(obj.get("id", f"line_{lineno:04d}")), "text": text.strip()})
        return items
    if in_path.suffix.lower() == ".txt":
        items.append({"id": in_path.stem, "text": in_path.read_text(encoding="utf-8-sig").strip()})
        return items
    die(f"unsupported input type: {in_path} (expected .jsonl, .txt or directory of .txt)")


def strip_think(text: str) -> str:
    text = THINK_RE.sub("", text, count=1)
    # Some decodes leave an empty/partial think block without a closing tag.
    text = re.sub(r"^\s*<think\b[^>]*>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def parse_full_output(raw: str) -> dict:
    text = strip_think(raw)
    doubtful: list = []
    edits: list = []
    parsed = None
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("text"), str):
            parsed = obj
            break
    if parsed is not None:
        text = parsed["text"]
        doubtful = parsed.get("doubtful") or []
        edits = parsed.get("edits") or []
        if not isinstance(doubtful, list):
            doubtful = [str(doubtful)]
        if not isinstance(edits, list):
            edits = []
        clean_edits = []
        for e in edits:
            if isinstance(e, dict) and "from" in e and "to" in e:
                clean_edits.append(
                    {"from": str(e["from"]), "to": str(e["to"]), "reason": str(e.get("reason", ""))}
                )
        edits = clean_edits
    # Inline markers are supported by eval_postproc; keep them in text and also
    # expose them as structured doubtful entries.
    for m in INLINE_DOUBTFUL_RE.finditer(text):
        parts = [p.strip() for p in m.group(1).split("|") if p.strip()]
        if parts:
            doubtful.append(parts[0] if len(parts) == 1 else {"span": parts[0], "suggestions": parts[1:]})
    return {"text": text.strip(), "doubtful": doubtful, "edits": edits}


def build_prompt(templates: dict, mode: str, text: str) -> str:
    sec = templates[mode]
    user = sec["user"].replace("{text}", text)
    return (
        f"<|im_start|>system\n{sec['system']}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def detect_backend(model_dir: Path, requested: str | None) -> str:
    if requested:
        return requested
    if (model_dir / "genai_config.json").is_file() and not (model_dir / "config.json").is_file():
        return "onnx"
    return "torch"


def run_torch(model_ref: str, device: str, items: list[dict], templates: dict, mode: str,
              max_new_tokens: int, threads: int) -> tuple[list[dict], dict]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        die(f"torch/transformers are not installed in this interpreter: {exc}")

    if device == "cuda" and not torch.cuda.is_available():
        die("CUDA requested but torch.cuda.is_available() is False (install a CUDA build of torch)")
    if threads:
        torch.set_num_threads(threads)

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"torch {torch.__version__} | device {device} | dtype {dtype}", file=sys.stderr)
    t_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    model = AutoModelForCausalLM.from_pretrained(model_ref, dtype=dtype, device_map=device)
    model.eval()
    load_s = time.perf_counter() - t_load
    print(f"model loaded in {load_s:.2f}s", file=sys.stderr)

    eos_ids = {151645, 151643}
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 151643

    results: list[dict] = []
    per_item: list[dict] = []
    t_start = time.perf_counter()
    for idx, item in enumerate(items, 1):
        prompt = build_prompt(templates, mode, item["text"])
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc.input_ids.to(device)
        attn = enc.attention_mask.to(device)
        prompt_tokens = int(input_ids.shape[1])
        # Safety cap: do not let the model continue an unfinished transcript.
        if mode == "full":
            cap = max(64, int(prompt_tokens * 3.0) + 32)
        else:
            cap = max(32, int(prompt_tokens * 1.6) + 16)
        item_max = min(max_new_tokens, cap)

        with torch.inference_mode():
            t0 = time.perf_counter()
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
            prefill_s = time.perf_counter() - t0
            past = out.past_key_values
            nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            gen_ids = [int(nxt.item())]
            t1 = time.perf_counter()
            for _ in range(item_max - 1):
                if gen_ids[-1] in eos_ids:
                    break
                attn = torch.cat(
                    [attn, torch.ones((1, 1), dtype=attn.dtype, device=attn.device)], dim=1
                )
                out = model(input_ids=nxt, attention_mask=attn, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                gen_ids.append(int(nxt.item()))
            decode_s = time.perf_counter() - t1

        raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
        if mode == "full":
            rec = parse_full_output(raw)
        else:
            rec = {"text": strip_think(raw), "doubtful": [], "edits": []}
        results.append({"id": item["id"], **rec})
        per_item.append(
            {
                "id": item["id"],
                "prompt_tokens": prompt_tokens,
                "generated_tokens": len(gen_ids),
                "prefill_s": round(prefill_s, 4),
                "decode_s": round(decode_s, 4),
                "total_s": round(prefill_s + decode_s, 4),
            }
        )
        print(
            f"[{idx}/{len(items)}] {item['id']}: {len(rec['text'])} chars, "
            f"{len(gen_ids)} tok, {prefill_s + decode_s:.2f}s",
            file=sys.stderr,
        )

    elapsed = time.perf_counter() - t_start
    gen_tokens = sum(p["generated_tokens"] for p in per_item)
    prefill_tokens = sum(p["prompt_tokens"] for p in per_item)
    prefill_s = sum(p["prefill_s"] for p in per_item)
    decode_s = sum(p["decode_s"] for p in per_item)
    metrics = {
        "backend": "torch",
        "device": device,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "model": model_ref,
        "mode": mode,
        "items": len(results),
        "load_s": round(load_s, 3),
        "wall_s": round(elapsed, 3),
        "sec_per_item": round(elapsed / max(1, len(results)), 4),
        "prompt_tokens": prefill_tokens,
        "generated_tokens": gen_tokens,
        "prefill_s": round(prefill_s, 3),
        "decode_s": round(decode_s, 3),
        "prefill_tokens_per_s": round(prefill_tokens / prefill_s, 2) if prefill_s else None,
        "decode_tokens_per_s": round(gen_tokens / decode_s, 2) if decode_s else None,
        "peak_cuda_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if device == "cuda" else None,
        "per_item": per_item,
    }
    return results, metrics


def run_onnx(model_dir: Path, device: str, items: list[dict], templates: dict, mode: str,
             max_new_tokens: int, threads: int) -> tuple[list[dict], dict]:
    if device == "cuda":
        die("onnx backend in this build exposes only CPUExecutionProvider; use --device cpu")
    if threads:
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
    try:
        import onnxruntime_genai as og
    except ImportError as exc:
        die(f"onnxruntime-genai is not installed in this interpreter: {exc}")

    print(f"onnxruntime-genai {og.__version__} | device cpu", file=sys.stderr)
    t_load = time.perf_counter()
    model = og.Model(str(model_dir))
    tokenizer = og.Tokenizer(model)
    load_s = time.perf_counter() - t_load
    print(f"model loaded in {load_s:.2f}s", file=sys.stderr)

    results: list[dict] = []
    per_item: list[dict] = []
    t_start = time.perf_counter()
    for idx, item in enumerate(items, 1):
        prompt = build_prompt(templates, mode, item["text"])
        ids = tokenizer.encode(prompt)
        if mode == "full":
            cap = max(64, int(len(ids) * 3.0) + 32)
        else:
            cap = max(32, int(len(ids) * 1.6) + 16)
        item_max = min(max_new_tokens, cap)
        max_length = len(ids) + item_max
        if max_length > 40960:
            die(f"item {item['id']}: prompt is too long ({len(ids)} tokens) for max_new_tokens={item_max}")
        params = og.GeneratorParams(model)
        params.set_search_options(
            do_sample=False,
            max_length=max_length,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            random_seed=42,
        )
        t0 = time.perf_counter()
        generator = og.Generator(model, params)
        generator.append_tokens(ids)
        out_ids: list[int] = []
        while not generator.is_done():
            generator.generate_next_token()
            out_ids.extend(generator.get_next_tokens())
        elapsed_item = time.perf_counter() - t0
        raw = tokenizer.decode(out_ids)
        if mode == "full":
            rec = parse_full_output(raw)
        else:
            rec = {"text": strip_think(raw), "doubtful": [], "edits": []}
        results.append({"id": item["id"], **rec})
        per_item.append(
            {
                "id": item["id"],
                "prompt_tokens": len(ids),
                "generated_tokens": len(out_ids),
                "total_s": round(elapsed_item, 4),
            }
        )
        print(
            f"[{idx}/{len(items)}] {item['id']}: {len(rec['text'])} chars, "
            f"{len(out_ids)} tok, {elapsed_item:.2f}s",
            file=sys.stderr,
        )

    elapsed = time.perf_counter() - t_start
    gen_tokens = sum(p["generated_tokens"] for p in per_item)
    metrics = {
        "backend": "onnx",
        "device": "cpu",
        "onnxruntime_genai_version": og.__version__,
        "model": str(model_dir),
        "mode": mode,
        "items": len(results),
        "load_s": round(load_s, 3),
        "wall_s": round(elapsed, 3),
        "sec_per_item": round(elapsed / max(1, len(results)), 4),
        "generated_tokens": gen_tokens,
        "decode_tokens_per_s": round(gen_tokens / elapsed, 2) if elapsed else None,
        "per_item": per_item,
    }
    return results, metrics


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Deterministic LLM post-processor for Russian ASR (torch or ONNX Runtime GenAI).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  .venv\\Scripts\\python.exe tools\\run_llm\\run_llm.py --model C:\\temp\\opencode\\qvikhr_src "
            "--in data\\postproc\\punct_eval.jsonl --out reports\\postproc_eval\\llm_punct.jsonl "
            "--mode punctuate --backend torch --device cuda\n"
            "  .venv\\Scripts\\python.exe tools\\run_llm\\run_llm.py "
            "--model models\\llm\\QVikhr-3-1.7B-Instruction-noreasoning-int4 "
            "--in data\\postproc\\punct_eval.jsonl --out reports\\postproc_eval\\llm_punct_onnx.jsonl "
            "--mode punctuate --backend onnx --device cpu --force\n"
        ),
    )
    ap.add_argument("--model", required=True, help="HF id or local model dir (torch) / ORT GenAI dir (onnx)")
    ap.add_argument("--in", dest="in_path", required=True, help="input .jsonl, .txt or directory of <id>.txt")
    ap.add_argument("--out", required=True, help="output .jsonl, .txt, or directory for <id>.txt")
    ap.add_argument("--mode", required=True, choices=["punctuate", "correct", "full"])
    ap.add_argument("--backend", choices=["torch", "onnx"], default=None,
                    help="inference backend (default: auto-detect from model dir)")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None,
                    help="device (default: cuda if available, else cpu)")
    ap.add_argument("--prompt-templates", default=str(DEFAULT_TEMPLATES), help="prompt template markdown file")
    ap.add_argument("--max-new-tokens", type=int, default=512, help="maximum generated tokens per item (default 512)")
    ap.add_argument("--threads", type=int, default=0, help="CPU thread count (0 = runtime default)")
    ap.add_argument("--metrics", default=None, help="optional path for a JSON metrics sidecar")
    ap.add_argument("--force", action="store_true", help="overwrite existing output")
    args = ap.parse_args()

    if args.max_new_tokens < 1:
        die("--max-new-tokens must be >= 1")
    if args.threads < 0:
        die("--threads must be >= 0")

    model_path = Path(args.model)
    backend = detect_backend(model_path, args.backend)
    if backend == "onnx" and not (model_path / "genai_config.json").is_file():
        die(f"not an ORT GenAI model directory (genai_config.json missing): {model_path}")
    if backend == "torch" and model_path.exists() and not (model_path / "config.json").is_file():
        die(f"not a transformers model directory (config.json missing): {model_path}")

    device = args.device
    if device is None:
        if backend == "onnx":
            device = "cpu"
        else:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

    templates = parse_templates(Path(args.prompt_templates))
    items = read_items(Path(args.in_path))
    if not items:
        die("input contains no items")

    out_path = Path(args.out)
    out_is_dir = out_path.is_dir() or args.out.endswith(("\\", "/"))
    if out_is_dir:
        out_path.mkdir(parents=True, exist_ok=True)
        existing = [out_path / f"{it['id']}.txt" for it in items if (out_path / f"{it['id']}.txt").exists()]
    else:
        if out_path.exists() and not args.force:
            die(f"output exists, use --force to overwrite: {out_path}", code=2)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        existing = [out_path] if out_path.exists() else []

    if existing and not args.force:
        die(f"output exists, use --force to overwrite: {existing[0]}", code=2)

    print(
        f"run_llm {TOOL_VERSION} | python {sys.version.split()[0]} | backend {backend} | device {device}",
        file=sys.stderr,
    )
    print(f"model: {args.model}", file=sys.stderr)
    print(f"mode: {args.mode} | items: {len(items)} | greedy | max_new_tokens: {args.max_new_tokens}", file=sys.stderr)

    if backend == "torch":
        results, metrics = run_torch(args.model, device, items, templates, args.mode,
                                     args.max_new_tokens, args.threads)
    else:
        results, metrics = run_onnx(model_path, device, items, templates, args.mode,
                                    args.max_new_tokens, args.threads)

    if out_is_dir:
        for rec in results:
            (out_path / f"{rec['id']}.txt").write_text(rec["text"], encoding="utf-8")
    elif out_path.suffix.lower() == ".jsonl":
        with out_path.open("w", encoding="utf-8") as fh:
            for rec in results:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        if len(results) == 1:
            out_path.write_text(results[0]["text"], encoding="utf-8")
        else:
            with out_path.open("w", encoding="utf-8") as fh:
                for rec in results:
                    fh.write(f"{rec['id']}\t{rec['text']}\n")

    if args.metrics:
        mp = Path(args.metrics)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"done: {len(results)} items in {metrics['wall_s']:.2f}s "
        f"({metrics['sec_per_item']:.3f}s/item) -> {out_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
