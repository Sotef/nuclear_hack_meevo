#!/usr/bin/env python3
"""run_asr — run an offline ASR backend over a manifest and save hypotheses.

Currently supports backend `sherpa-onnx` (offline transducer, int8/fp32).
Writes <out>/<id>.txt, <out>/timestamps.json (token/word/segment times) and
<out>/metrics.json (wall time, RTF, peak RSS). Deterministic greedy decoding.

Usage:
    python tools/run_asr/run_asr.py --model models/sherpa-onnx-zipformer-ru-2024-09-18 \
        --manifest data/test/manifest.jsonl --out tools/eval_asr/hyp/zipformer_ru \
        --backend sherpa-onnx --precision int8 --repeat 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

TOOL_NAME = "run_asr"
TOOL_VERSION = "1.0.0"


def load_manifest(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on manifest line {lineno}: {exc}") from exc
            for key in ("id", "path"):
                if key not in rec:
                    raise ValueError(f"manifest line {lineno} missing field '{key}'")
            records.append(rec)
    if not records:
        raise ValueError("manifest is empty")
    return records


def resolve_audio(value: str, manifest_dir: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    candidate = manifest_dir / p
    if candidate.exists():
        return candidate
    return Path.cwd() / p


def pick(model_dir: Path, stem: str, precision: str) -> Path:
    if precision == "int8":
        p = model_dir / f"{stem}.int8.onnx"
        if p.exists():
            return p
    p = model_dir / f"{stem}.onnx"
    if p.exists():
        return p
    raise FileNotFoundError(f"no {stem}.onnx / {stem}.int8.onnx in {model_dir}")


def build_recognizer(model_dir: Path, precision: str, threads: int):
    import sherpa_onnx

    encoder = pick(model_dir, "encoder", precision)
    decoder = pick(model_dir, "decoder", precision)
    joiner = pick(model_dir, "joiner", precision)
    tokens = model_dir / "tokens.txt"
    if not tokens.exists():
        raise FileNotFoundError(f"tokens.txt not found in {model_dir}")
    kwargs = dict(
        encoder=str(encoder),
        decoder=str(decoder),
        joiner=str(joiner),
        tokens=str(tokens),
        num_threads=threads,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        provider="cpu",
    )
    bpe = model_dir / "bpe.model"
    if bpe.exists():
        kwargs["modeling_unit"] = "bpe"
        kwargs["bpe_vocab"] = str(bpe)
    return sherpa_onnx.OfflineRecognizer.from_transducer(**kwargs)


def read_wav(path: Path):
    import soundfile as sf

    samples, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"{path}: expected 16000 Hz, got {sr}")
    return samples


def tokens_to_words(tokens: list[str], timestamps: list[float]) -> list[dict]:
    """Group BPE tokens into words. A new word starts on a token with a leading space
    or on the sentencepiece marker (▁)."""
    words: list[dict] = []
    cur_text = ""
    cur_start = None
    cur_end = None
    for tok, ts in zip(tokens, timestamps):
        is_new = tok.startswith(" ") or tok.startswith("\u2581")
        clean = tok.replace("\u2581", " ").strip()
        if not clean:
            continue
        if is_new and cur_text:
            words.append({"text": cur_text, "start": round(cur_start, 3), "end": round(cur_end, 3)})
            cur_text, cur_start, cur_end = "", None, None
        if cur_start is None:
            cur_start = ts
        cur_text = clean if (is_new or not cur_text) else cur_text + clean
        cur_end = ts
    if cur_text:
        words.append({"text": cur_text, "start": round(cur_start, 3), "end": round(cur_end, 3)})
    return words


def words_to_segments(words: list[dict], duration: float, gap: float) -> list[dict]:
    """Heuristic segmentation: split between words separated by >= gap seconds."""
    if not words:
        return []
    segments = []
    seg_start = words[0]["start"]
    seg_words = [words[0]["text"]]
    prev_end = words[0]["end"]
    for w in words[1:]:
        if w["start"] - prev_end >= gap:
            segments.append({"start": round(seg_start, 3), "end": round(prev_end, 3),
                             "text": " ".join(seg_words)})
            seg_start = w["start"]
            seg_words = [w["text"]]
        else:
            seg_words.append(w["text"])
        prev_end = w["end"]
    segments.append({"start": round(seg_start, 3), "end": round(prev_end, 3),
                     "text": " ".join(seg_words)})
    return segments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_asr",
        description="Run an offline ASR backend over a manifest; save hypotheses, timestamps, metrics.",
        epilog=(
            "Example: python tools/run_asr/run_asr.py "
            "--model models/sherpa-onnx-zipformer-ru-2024-09-18 "
            "--manifest data/test/manifest.jsonl --out tools/eval_asr/hyp/zipformer_ru "
            "--backend sherpa-onnx --precision int8 --repeat 2"
        ),
    )
    parser.add_argument("--model", required=True, type=Path, help="model directory")
    parser.add_argument("--manifest", required=True, type=Path, help="manifest.jsonl")
    parser.add_argument("--out", required=True, type=Path, help="output directory for <id>.txt")
    parser.add_argument("--backend", default="sherpa-onnx", choices=["sherpa-onnx"],
                        help="ASR backend (default: sherpa-onnx)")
    parser.add_argument("--precision", default="int8", choices=["int8", "fp32"],
                        help="model precision to load (default: int8)")
    parser.add_argument("--threads", type=int, default=1, help="inference threads (default: 1)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="number of full inference passes for timing (default: 1)")
    parser.add_argument("--segment-gap", type=float, default=0.5,
                        help="gap in seconds that splits heuristic segments (default: 0.5)")
    parser.add_argument("--max-chunk-sec", type=float, default=20.0,
                        help="split audio into chunks of at most this many seconds before decoding "
                             "(zipformer-ru-2024-09-18 fails above ~20 s; 0 disables). Default: 20")
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = parser.parse_args(argv)

    try:
        records = load_manifest(args.manifest)
        manifest_dir = args.manifest.resolve().parent
        audio = [(r["id"], resolve_audio(r["path"], manifest_dir), r.get("duration_sec")) for r in records]
        for rid, p, _ in audio:
            if not p.exists():
                raise FileNotFoundError(f"audio not found for {rid}: {p}")
        if args.out.exists() and any(args.out.glob("*.txt")) and not args.force:
            raise FileExistsError(f"{args.out} already has .txt files; use --force to overwrite")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    import psutil

    proc = psutil.Process()
    peak_rss = proc.memory_info().rss

    t0 = time.perf_counter()
    recognizer = build_recognizer(args.model, args.precision, args.threads)
    load_sec = time.perf_counter() - t0

    args.out.mkdir(parents=True, exist_ok=True)
    total_audio = 0.0
    pass_times: list[float] = []
    first_pass: dict[str, dict] = {}

    for pass_idx in range(args.repeat):
        pass_start = time.perf_counter()
        for rid, path, dur in audio:
            samples = read_wav(path)
            audio_dur = len(samples) / 16000.0
            if pass_idx == 0:
                total_audio += audio_dur
            t_inf = time.perf_counter()
            chunk_len = int(args.max_chunk_sec * 16000) if args.max_chunk_sec > 0 else len(samples)
            chunk_len = max(chunk_len, 16000)
            texts: list[str] = []
            all_tokens: list[dict] = []
            all_words: list[dict] = []
            for off in range(0, len(samples), chunk_len):
                chunk = samples[off:off + chunk_len]
                if len(chunk) < 1600:  # < 0.1 s tail
                    continue
                stream = recognizer.create_stream()
                stream.accept_waveform(16000, chunk)
                recognizer.decode_stream(stream)
                res = stream.result
                offset = off / 16000.0
                texts.append(res.text.strip())
                toks = list(res.tokens)
                tss = [float(t) + offset for t in res.timestamps]
                all_tokens.extend({"token": t, "start": round(float(s), 3)} for t, s in zip(toks, tss))
                all_words.extend(tokens_to_words(toks, tss))
            inf_sec = time.perf_counter() - t_inf
            peak_rss = max(peak_rss, proc.memory_info().rss)
            if pass_idx == 0:
                text = " ".join(t for t in texts if t).strip()
                segments = words_to_segments(all_words, audio_dur, args.segment_gap)
                first_pass[rid] = {
                    "duration_sec": round(audio_dur, 3),
                    "text": text,
                    "tokens": all_tokens,
                    "words": all_words,
                    "segments": segments,
                    "inference_sec": round(inf_sec, 4),
                }
                (args.out / f"{rid}.txt").write_text(text + "\n", encoding="utf-8")
        pass_times.append(time.perf_counter() - pass_start)

    avg_pass = sum(pass_times) / len(pass_times)
    rtf = avg_pass / total_audio if total_audio else 0.0
    metrics = {
        "tool": TOOL_NAME,
        "version": TOOL_VERSION,
        "backend": args.backend,
        "model_dir": str(args.model),
        "precision": args.precision,
        "threads": args.threads,
        "decoding_method": "greedy_search",
        "records": len(audio),
        "total_audio_sec": round(total_audio, 3),
        "model_load_sec": round(load_sec, 4),
        "passes": [round(t, 4) for t in pass_times],
        "avg_pass_sec": round(avg_pass, 4),
        "rtf": round(rtf, 6),
        "peak_rss_mb": round(peak_rss / (1024 * 1024), 2),
        "per_record": {rid: {"duration_sec": d["duration_sec"],
                             "inference_sec": d["inference_sec"],
                             "rtf": round(d["inference_sec"] / d["duration_sec"], 6) if d["duration_sec"] else None}
                       for rid, d in first_pass.items()},
    }
    (args.out / "timestamps.json").write_text(
        json.dumps(first_pass, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{TOOL_NAME} {TOOL_VERSION} backend={args.backend} precision={args.precision} threads={args.threads}")
    print(f"records={len(audio)} audio={total_audio:.3f}s load={load_sec:.3f}s "
          f"passes={metrics['passes']} avg={avg_pass:.3f}s RTF={rtf:.4f} peak_rss={metrics['peak_rss_mb']:.1f}MB")
    print(f"out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
