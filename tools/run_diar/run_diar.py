#!/usr/bin/env python3
"""run_diar — offline speaker diarization baseline with sherpa-onnx.

Pipeline: pyannote segmentation-3.0 (ONNX) -> speaker embeddings (CAM++ / TitaNet)
-> FastClustering -> RTTM hypotheses + JSON run report (wall time, RTF, #speakers).

Usage:
    python tools/run_diar/run_diar.py --manifest data/test/manifest.jsonl \
        --out reports/diar_engineer/hyp/A_thr0.5 --threshold 0.5 \
        --report reports/diar_engineer/run_A.json
    python tools/run_diar/run_diar.py --manifest data/test/manifest.jsonl \
        --out reports/diar_engineer/hyp/B_oracle --oracle-num-speakers \
        --report reports/diar_engineer/run_B.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

TOOL_NAME = "run_diar"
TOOL_VERSION = "1.0.0"

DEFAULT_SEG = "models/diar/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
DEFAULT_EMB = "models/diar/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: bad JSON: {exc}") from exc
    if not rows:
        raise ValueError(f"{path}: empty manifest")
    return rows


def build_diarizer(args, num_clusters: int = -1):
    import sherpa_onnx as so

    seg = so.OfflineSpeakerSegmentationModelConfig()
    seg.pyannote.model = str(args.seg_model)
    seg.num_threads = args.num_threads
    seg.provider = args.provider

    emb = so.SpeakerEmbeddingExtractorConfig()
    emb.model = str(args.embedding_model)
    emb.num_threads = args.num_threads
    emb.provider = args.provider

    clu = so.FastClusteringConfig()
    clu.threshold = args.threshold
    clu.num_clusters = num_clusters
    clu.compute_confidence = False

    cfg = so.OfflineSpeakerDiarizationConfig(
        segmentation=seg,
        embedding=emb,
        clustering=clu,
        min_duration_on=args.min_duration_on,
        min_duration_off=args.min_duration_off,
    )
    return so.OfflineSpeakerDiarization(cfg), so


def write_rttm(path: Path, file_id: str, segments) -> None:
    lines = []
    for seg in segments:
        dur = seg.end - seg.start
        if dur <= 0:
            continue
        lines.append(
            f"SPEAKER {file_id} 1 {seg.start:.3f} {dur:.3f} <NA> <NA> "
            f"SPEAKER_{seg.speaker + 1} <NA> <NA>"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_diar",
        description="Offline diarization baseline (sherpa-onnx pyannote + embeddings + FastClustering).",
        epilog=(
            "Example: python tools/run_diar/run_diar.py --manifest data/test/manifest.jsonl "
            "--out hyp/A --threshold 0.5 --report run_A.json"
        ),
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="output dir for <id>.rttm")
    parser.add_argument("--report", type=Path, help="optional JSON run report path")
    parser.add_argument("--seg-model", type=Path, default=Path(DEFAULT_SEG))
    parser.add_argument("--embedding-model", type=Path, default=Path(DEFAULT_EMB))
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="FastClustering cosine threshold (used when not oracle)")
    parser.add_argument("--oracle-num-speakers", action="store_true",
                        help="use manifest speakers_count as num_clusters per file")
    parser.add_argument("--num-speakers", type=int, default=-1,
                        help="fixed num_clusters for all files (overrides threshold)")
    parser.add_argument("--min-duration-on", type=float, default=0.3)
    parser.add_argument("--min-duration-off", type=float, default=0.5)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--ids", nargs="*", help="optional subset of manifest ids")
    parser.add_argument("--force", action="store_true", help="overwrite existing RTTM files")
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    for p in (args.seg_model, args.embedding_model):
        if not p.is_file():
            print(f"ERROR: model not found: {p}", file=sys.stderr)
            return 2

    try:
        rows = load_manifest(args.manifest)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.ids:
        wanted = set(args.ids)
        rows = [r for r in rows if r["id"] in wanted]
        if not rows:
            print("ERROR: --ids matched no manifest records", file=sys.stderr)
            return 2

    args.out.mkdir(parents=True, exist_ok=True)
    existing = [args.out / f"{r['id']}.rttm" for r in rows if (args.out / f"{r['id']}.rttm").exists()]
    if existing and not args.force:
        print("ERROR: output RTTM(s) already exist; use --force:", file=sys.stderr)
        for p in existing:
            print(f"  {p}", file=sys.stderr)
        return 2

    try:
        _, so = build_diarizer(args)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: failed to init sherpa-onnx diarizer: {exc}", file=sys.stderr)
        return 2

    manifest_dir = args.manifest.parent
    results = []
    total_wall = 0.0
    total_dur = 0.0
    for row in rows:
        file_id = row["id"]
        wav_path = Path(row["path"])
        if not wav_path.is_absolute():
            wav_path = manifest_dir / wav_path
        if not wav_path.is_file():
            print(f"ERROR: audio not found for {file_id}: {wav_path}", file=sys.stderr)
            return 2
        samples, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if sr != 16000:
            print(f"ERROR: {file_id}: expected 16 kHz, got {sr}", file=sys.stderr)
            return 2
        duration = len(samples) / sr

        # Per-file clustering config: oracle/fixed num_clusters vs threshold.
        if args.oracle_num_speakers:
            n = int(row.get("speakers_count", -1))
            if n < 1:
                print(f"ERROR: {file_id}: no speakers_count in manifest for oracle mode", file=sys.stderr)
                return 2
        elif args.num_speakers > 0:
            n = args.num_speakers
        else:
            n = -1
        diarizer, _ = build_diarizer(args, num_clusters=n)

        t0 = time.perf_counter()
        result = diarizer.process(samples)
        wall = time.perf_counter() - t0
        segments = result.sort_by_start_time()
        write_rttm(args.out / f"{file_id}.rttm", file_id, segments)

        total_wall += wall
        total_dur += duration
        rec = {
            "id": file_id,
            "duration_sec": round(duration, 3),
            "wall_sec": round(wall, 3),
            "rtf": round(wall / duration, 4) if duration else None,
            "num_speakers_hyp": int(result.num_speakers),
            "num_speakers_ref": row.get("speakers_count"),
            "num_segments": int(result.num_segments),
            "clustering": "oracle" if args.oracle_num_speakers else (
                f"fixed:{args.num_speakers}" if args.num_speakers > 0 else f"threshold:{args.threshold}"
            ),
        }
        results.append(rec)
        print(f"{file_id}: dur={duration:.2f}s wall={wall:.2f}s rtf={rec['rtf']} "
              f"spk={rec['num_speakers_hyp']} (ref={rec['num_speakers_ref']}) segs={rec['num_segments']}")

    payload = {
        "tool": TOOL_NAME,
        "version": TOOL_VERSION,
        "manifest": str(args.manifest),
        "out": str(args.out),
        "seg_model": str(args.seg_model),
        "embedding_model": str(args.embedding_model),
        "threshold": args.threshold,
        "oracle_num_speakers": args.oracle_num_speakers,
        "num_speakers": args.num_speakers,
        "min_duration_on": args.min_duration_on,
        "min_duration_off": args.min_duration_off,
        "num_threads": args.num_threads,
        "provider": args.provider,
        "sherpa_onnx_version": getattr(so, "__version__", "unknown"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "soundfile": sf.__version__,
        "total_duration_sec": round(total_dur, 3),
        "total_wall_sec": round(total_wall, 3),
        "overall_rtf": round(total_wall / total_dur, 4) if total_dur else None,
        "files": results,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"report={args.report}")
    print(f"TOTAL dur={total_dur:.2f}s wall={total_wall:.2f}s RTF={payload['overall_rtf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
