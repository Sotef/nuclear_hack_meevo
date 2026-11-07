#!/usr/bin/env python3
"""eval_diar — frame-based DER for RTTM reference/hypothesis files.

Implements NIST-style DER with a collar around reference speaker boundaries:
    DER = (miss + false_alarm + speaker_error) / total_reference_speech
Reports overall, per-file, by reference speaker count, and by overlap/non-overlap.

Usage:
    python tools/eval_diar/eval_diar.py --ref ref.rttm --hyp hyp.rttm \
        --collar 0.25 --report reports/data_eval/der_report
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

TOOL_NAME = "eval_diar"
TOOL_VERSION = "1.0.1"
FRAME_STEP = 0.01  # 10 ms


def parse_rttm(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    segments: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] != "SPEAKER":
                continue
            if len(parts) < 8:
                raise ValueError(f"{path}:{lineno}: malformed RTTM line")
            try:
                start = float(parts[3])
                dur = float(parts[4])
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: bad start/duration") from exc
            if dur <= 0:
                continue
            segments[parts[1]].append((start, start + dur, parts[7]))
    if not segments:
        raise ValueError(f"no SPEAKER segments found in {path}")
    return dict(segments)


def collect_pairs(ref_path: Path, hyp_path: Path) -> list[tuple[str, Path, Path]]:
    if ref_path.is_dir() and hyp_path.is_dir():
        refs = {p.stem: p for p in ref_path.glob("*.rttm")}
        hyps = {p.stem: p for p in hyp_path.glob("*.rttm")}
        common = sorted(set(refs) & set(hyps))
        if not common:
            raise FileNotFoundError("no matching *.rttm basenames in ref/hyp directories")
        return [(name, refs[name], hyps[name]) for name in common]
    if ref_path.is_file() and hyp_path.is_file():
        return [(ref_path.stem, ref_path, hyp_path)]
    raise ValueError("--ref and --hyp must both be files or both be directories")


def active_at(segments: list[tuple[float, float, str]], t: float) -> set[str]:
    return {spk for start, end, spk in segments if start <= t < end}


def in_collar(t: float, segments: list[tuple[float, float, str]], collar: float) -> bool:
    if collar <= 0:
        return False
    for start, end, _ in segments:
        if abs(t - start) <= collar or abs(t - end) <= collar:
            return True
    return False


def best_mapping(cooc: dict[str, dict[str, int]], ref_spks: list[str], hyp_spks: list[str]) -> dict[str, str]:
    """Maximize co-occurrence. Brute force for <=7 hyp speakers, greedy otherwise."""
    if not ref_spks or not hyp_spks:
        return {}
    if len(hyp_spks) <= 7:
        best_score = -1
        best_perm = None
        for perm in itertools.permutations(hyp_spks, min(len(ref_spks), len(hyp_spks))):
            score = sum(cooc.get(r, {}).get(h, 0) for r, h in zip(ref_spks, perm))
            if score > best_score:
                best_score = score
                best_perm = perm
        mapping = {}
        if best_perm:
            for r, h in zip(ref_spks, best_perm):
                mapping[r] = h
        return mapping
    mapping = {}
    used = set()
    for r in sorted(ref_spks, key=lambda x: -sum(cooc.get(x, {}).values())):
        candidates = [(cooc.get(r, {}).get(h, 0), h) for h in hyp_spks if h not in used]
        if candidates:
            _, h = max(candidates)
            mapping[r] = h
            used.add(h)
    return mapping


def evaluate_file(
    ref_segs: list[tuple[float, float, str]],
    hyp_segs: list[tuple[float, float, str]],
    collar: float,
) -> dict:
    end_time = max([e for _, e, _ in ref_segs] + [e for _, e, _ in hyp_segs])
    n_frames = int(round(end_time / FRAME_STEP)) + 1

    ref_spks = sorted({s for _, _, s in ref_segs})
    hyp_spks = sorted({s for _, _, s in hyp_segs})

    # Co-occurrence for speaker mapping (only non-collar frames with both active).
    cooc: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for k in range(n_frames):
        t = k * FRAME_STEP
        if in_collar(t, ref_segs, collar):
            continue
        ra = active_at(ref_segs, t)
        ha = active_at(hyp_segs, t)
        if ra and ha:
            for r in ra:
                for h in ha:
                    cooc[r][h] += 1
    mapping = best_mapping(cooc, ref_spks, hyp_spks)

    stats = {
        "total_ref": 0, "miss": 0, "false_alarm": 0, "speaker_error": 0,
        "overlap_total_ref": 0, "overlap_miss": 0, "overlap_false_alarm": 0,
        "overlap_speaker_error": 0,
        "nonoverlap_total_ref": 0, "nonoverlap_miss": 0,
        "nonoverlap_false_alarm": 0, "nonoverlap_speaker_error": 0,
    }
    for k in range(n_frames):
        t = k * FRAME_STEP
        if in_collar(t, ref_segs, collar):
            continue
        ra = active_at(ref_segs, t)
        ha = active_at(hyp_segs, t)
        if not ra and not ha:
            continue
        is_overlap = len(ra) > 1
        # Speaker-time accounting: overlapping frames count once per active speaker.
        if ra:
            stats["total_ref"] += len(ra)
            if is_overlap:
                stats["overlap_total_ref"] += len(ra)
            else:
                stats["nonoverlap_total_ref"] += len(ra)
        if ra and not ha:
            stats["miss"] += len(ra)
            if is_overlap:
                stats["overlap_miss"] += len(ra)
            else:
                stats["nonoverlap_miss"] += len(ra)
        elif ha and not ra:
            stats["false_alarm"] += len(ha)
            if is_overlap:
                stats["overlap_false_alarm"] += len(ha)
            else:
                stats["nonoverlap_false_alarm"] += len(ha)
        elif ra and ha:
            # Ref speech is covered by hyp speech; count misassigned ref speakers.
            for r in ra:
                if mapping.get(r) not in ha:
                    stats["speaker_error"] += 1
                    if is_overlap:
                        stats["overlap_speaker_error"] += 1
                    else:
                        stats["nonoverlap_speaker_error"] += 1

    def der(miss: int, fa: int, spk: int, total: int) -> float:
        return ((miss + fa + spk) / total) if total else 0.0

    stats["der"] = der(stats["miss"], stats["false_alarm"], stats["speaker_error"], stats["total_ref"])
    stats["overlap_der"] = der(
        stats["overlap_miss"], stats["overlap_false_alarm"],
        stats["overlap_speaker_error"], stats["overlap_total_ref"],
    )
    stats["nonoverlap_der"] = der(
        stats["nonoverlap_miss"], stats["nonoverlap_false_alarm"],
        stats["nonoverlap_speaker_error"], stats["nonoverlap_total_ref"],
    )
    stats["ref_speakers"] = len(ref_spks)
    stats["hyp_speakers"] = len(hyp_spks)
    stats["mapping"] = mapping
    stats["duration_sec"] = round(end_time, 3)
    return stats


def write_report(base: Path, payload: dict, text: str) -> tuple[Path, Path]:
    if base.suffix.lower() == ".json":
        json_path = base
        txt_path = base.with_suffix(".txt")
    else:
        json_path = base.with_suffix(".json")
        txt_path = base.with_suffix(".txt")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    txt_path.write_text(text, encoding="utf-8")
    return json_path, txt_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval_diar",
        description="Frame-based DER for RTTM reference/hypothesis files (collar, overlap breakdown).",
        epilog=(
            "Example: python tools/eval_diar/eval_diar.py "
            "--ref data/test/diar_0001.rttm --hyp hyp/diar_0001.rttm "
            "--collar 0.25 --report reports/data_eval/der_report"
        ),
    )
    parser.add_argument("--ref", required=True, type=Path, help="reference RTTM file or directory")
    parser.add_argument("--hyp", required=True, type=Path, help="hypothesis RTTM file or directory")
    parser.add_argument("--collar", type=float, default=0.25, help="collar in seconds (default: 0.25)")
    parser.add_argument("--report", required=True, type=Path,
                        help="output base path; writes <base>.json and <base>.txt")
    args = parser.parse_args(argv)

    if args.collar < 0:
        print("ERROR: --collar must be >= 0", file=sys.stderr)
        return 2

    try:
        pairs = collect_pairs(args.ref, args.hyp)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    per_file = []
    agg = {
        "total_ref": 0, "miss": 0, "false_alarm": 0, "speaker_error": 0,
        "overlap_total_ref": 0, "overlap_miss": 0, "overlap_false_alarm": 0,
        "overlap_speaker_error": 0,
        "nonoverlap_total_ref": 0, "nonoverlap_miss": 0,
        "nonoverlap_false_alarm": 0, "nonoverlap_speaker_error": 0,
    }
    by_speakers: dict[int, dict] = defaultdict(lambda: {
        "files": 0, "total_ref": 0, "miss": 0, "false_alarm": 0, "speaker_error": 0,
    })

    for name, ref_path, hyp_path in pairs:
        try:
            ref_segs = parse_rttm(ref_path)
            hyp_segs = parse_rttm(hyp_path)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        # Evaluate each file present in the reference; missing hyp file = all miss.
        for fname, rsegs in sorted(ref_segs.items()):
            hsegs = hyp_segs.get(fname, [])
            st = evaluate_file(rsegs, hsegs, args.collar)
            st["file"] = fname
            st["ref_rttm"] = str(ref_path)
            st["hyp_rttm"] = str(hyp_path)
            per_file.append(st)
            for key in agg:
                agg[key] += st[key]
            bucket = by_speakers[st["ref_speakers"]]
            bucket["files"] += 1
            for key in ("total_ref", "miss", "false_alarm", "speaker_error"):
                bucket[key] += st[key]

    def der(miss: int, fa: int, spk: int, total: int) -> float:
        return ((miss + fa + spk) / total) if total else 0.0

    summary = {
        "files": len(per_file),
        "collar_sec": args.collar,
        "frame_step_sec": FRAME_STEP,
        "units": "speaker-time frames at 10 ms (overlaps counted per active speaker)",
        "total_ref_frames": agg["total_ref"],
        "miss": agg["miss"],
        "false_alarm": agg["false_alarm"],
        "speaker_error": agg["speaker_error"],
        "der": round(der(agg["miss"], agg["false_alarm"], agg["speaker_error"], agg["total_ref"]), 6),
        "overlap": {
            "total_ref_frames": agg["overlap_total_ref"],
            "miss": agg["overlap_miss"],
            "false_alarm": agg["overlap_false_alarm"],
            "speaker_error": agg["overlap_speaker_error"],
            "der": round(der(agg["overlap_miss"], agg["overlap_false_alarm"],
                             agg["overlap_speaker_error"], agg["overlap_total_ref"]), 6),
        },
        "nonoverlap": {
            "total_ref_frames": agg["nonoverlap_total_ref"],
            "miss": agg["nonoverlap_miss"],
            "false_alarm": agg["nonoverlap_false_alarm"],
            "speaker_error": agg["nonoverlap_speaker_error"],
            "der": round(der(agg["nonoverlap_miss"], agg["nonoverlap_false_alarm"],
                             agg["nonoverlap_speaker_error"], agg["nonoverlap_total_ref"]), 6),
        },
        "by_ref_speaker_count": {
            str(k): {
                "files": v["files"],
                "total_ref_frames": v["total_ref"],
                "miss": v["miss"],
                "false_alarm": v["false_alarm"],
                "speaker_error": v["speaker_error"],
                "der": round(der(v["miss"], v["false_alarm"], v["speaker_error"], v["total_ref"]), 6),
            }
            for k, v in sorted(by_speakers.items())
        },
    }
    payload = {
        "tool": TOOL_NAME,
        "version": TOOL_VERSION,
        "ref": str(args.ref),
        "hyp": str(args.hyp),
        "summary": summary,
        "files": per_file,
    }

    lines = [
        f"{TOOL_NAME} {TOOL_VERSION}",
        f"ref: {args.ref}",
        f"hyp: {args.hyp}",
        f"collar: {args.collar}s  frame_step: {FRAME_STEP}s",
        "",
        f"{'file':<28} {'spk':>4} {'ref_spk_fr':>10} {'miss':>6} {'fa':>6} {'spk_err':>8} {'DER':>8}",
        "-" * 76,
    ]
    for st in per_file:
        lines.append(
            f"{st['file']:<28} {st['ref_speakers']:>4} {st['total_ref']:>8} "
            f"{st['miss']:>6} {st['false_alarm']:>6} {st['speaker_error']:>8} "
            f"{st['der']:>8.4f}"
        )
    lines += [
        "-" * 76,
        f"OVERALL DER = {summary['der']:.6f}  "
        f"(miss={summary['miss']} fa={summary['false_alarm']} "
        f"spk_err={summary['speaker_error']} ref_speaker_frames={summary['total_ref_frames']})",
        f"OVERLAP DER = {summary['overlap']['der']:.6f}  "
        f"(ref_speaker_frames={summary['overlap']['total_ref_frames']})",
        f"NON-OVERLAP DER = {summary['nonoverlap']['der']:.6f}  "
        f"(ref_speaker_frames={summary['nonoverlap']['total_ref_frames']})",
        "",
        "by reference speaker count:",
    ]
    for k, v in summary["by_ref_speaker_count"].items():
        lines.append(f"  {k} speaker(s): files={v['files']} DER={v['der']:.6f} "
                     f"ref_speaker_frames={v['total_ref_frames']}")
    lines += ["", f"dependencies: python={sys.version.split()[0]} (no external runtime deps)"]
    text = "\n".join(lines) + "\n"

    json_path, txt_path = write_report(args.report, payload, text)
    print(text)
    print(f"report_json={json_path}")
    print(f"report_txt={txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
