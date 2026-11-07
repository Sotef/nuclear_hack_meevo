#!/usr/bin/env python3
"""prep_audio — convert audio files to mono WAV at a target sample rate.

Usage:
    python tools/prep_audio/prep_audio.py INPUT OUTPUT [--sample-rate 16000] [--force]

INPUT may be a single audio file or a directory (non-recursive).
OUTPUT is a directory. Output file names are <input-stem>.wav.
Existing outputs are never overwritten unless --force is given.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

TOOL_NAME = "prep_audio"
TOOL_VERSION = "1.0.0"
AUDIO_EXTS = {
    ".wav", ".flac", ".ogg", ".oga", ".mp3", ".m4a", ".aac",
    ".opus", ".wma", ".aiff", ".aif", ".au",
}


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear-interpolation resampler (adequate for eval prep, not studio quality)."""
    if sr_in == sr_out or len(x) == 0:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    if n_out <= 0:
        return x
    t_in = np.arange(len(x), dtype=np.float64)
    t_out = np.linspace(0.0, len(x) - 1, n_out, dtype=np.float64)
    return np.interp(t_out, t_in, x).astype(np.float32)


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if data.size == 0:
        raise ValueError("audio stream is empty")
    mono = data.mean(axis=1)
    return mono, int(sr)


def collect_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        )
        if not files:
            raise FileNotFoundError(
                f"no supported audio files in directory: {input_path}"
            )
        return files
    raise FileNotFoundError(f"input does not exist: {input_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prep_audio",
        description="Convert audio to mono WAV at a target sample rate.",
        epilog=(
            "Example: python tools/prep_audio/prep_audio.py "
            "data/raw/extracted data/test --sample-rate 16000"
        ),
    )
    parser.add_argument("input", type=Path, help="input audio file or directory")
    parser.add_argument("output", type=Path, help="output directory")
    parser.add_argument(
        "--sample-rate", type=int, default=16000,
        help="target sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite existing output files",
    )
    args = parser.parse_args(argv)

    if args.sample_rate <= 0:
        print("ERROR: --sample-rate must be a positive integer", file=sys.stderr)
        return 2

    try:
        inputs = collect_inputs(args.input)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    out_dir: Path = args.output
    if out_dir.exists() and not out_dir.is_dir():
        print(f"ERROR: output path exists and is not a directory: {out_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[Path, Path]] = []
    for src in inputs:
        dst = out_dir / (src.stem + ".wav")
        jobs.append((src, dst))

    if not args.force:
        existing = [str(dst) for _, dst in jobs if dst.exists()]
        if existing:
            print(
                "ERROR: output file(s) already exist; refusing to overwrite "
                "without --force:\n  " + "\n  ".join(existing),
                file=sys.stderr,
            )
            return 2

    print(f"{TOOL_NAME} {TOOL_VERSION}")
    print(f"soundfile={sf.__version__} numpy={np.__version__}")
    print(f"input={args.input} files={len(jobs)} output={out_dir} "
          f"sample_rate={args.sample_rate} force={args.force}")

    failures = 0
    total_dur = 0.0
    for src, dst in jobs:
        try:
            mono, sr_in = load_mono(src)
            mono = resample_linear(mono, sr_in, args.sample_rate)
            mono = np.clip(mono, -1.0, 1.0)
            sf.write(str(dst), mono, args.sample_rate, subtype="PCM_16")
            dur = len(mono) / args.sample_rate
            total_dur += dur
            print(f"OK   {src.name} -> {dst.name}  "
                  f"in_sr={sr_in} out_sr={args.sample_rate} dur={dur:.3f}s")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {src}: {exc}", file=sys.stderr)

    print(f"done: ok={len(jobs) - failures} failed={failures} "
          f"total_duration={total_dur:.3f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
