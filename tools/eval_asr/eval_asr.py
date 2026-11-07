#!/usr/bin/env python3
"""eval_asr — word-level WER for Russian ASR hypotheses.

Normalization: lowercase, yo -> ye, punctuation removed, whitespace collapsed.
Hypotheses are read from a directory of <id>.txt files, a single .txt file
(applied to every record), or a JSON mapping id -> text.

Usage:
    python tools/eval_asr/eval_asr.py --manifest data/test/manifest.jsonl \
        --hyp hyp_dir --report reports/data_eval/asr_report
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TOOL_NAME = "eval_asr"
TOOL_VERSION = "1.0.0"


def normalize_ru(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def words(text: str) -> list[str]:
    norm = normalize_ru(text)
    return norm.split() if norm else []


def align_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Return (substitutions, deletions, insertions) for the optimal alignment."""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        ri = ref[i - 1]
        for j in range(1, n + 1):
            if ri == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    s = d = ins = 0
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            s += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, d, ins


def resolve_path(value: str, manifest_dir: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    candidate = manifest_dir / p
    if candidate.exists():
        return candidate
    return Path.cwd() / p


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
            for key in ("id", "path", "text_path"):
                if key not in rec:
                    raise ValueError(f"manifest line {lineno} missing field '{key}'")
            records.append(rec)
    if not records:
        raise ValueError("manifest is empty")
    return records


def load_hypotheses(args, records: list[dict], manifest_dir: Path) -> dict[str, str]:
    ids = [r["id"] for r in records]
    if args.hyp_json:
        data = json.loads(Path(args.hyp_json).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "hypotheses" in data and isinstance(data["hypotheses"], dict):
                data = data["hypotheses"]
            return {str(k): str(v) for k, v in data.items()}
        if isinstance(data, list):
            out = {}
            for item in data:
                if not isinstance(item, dict) or "id" not in item:
                    raise ValueError("--hyp-json list items must be objects with 'id' and 'text'")
                out[str(item["id"])] = str(item.get("text", ""))
            return out
        raise ValueError("--hyp-json must be a mapping or a list of {id,text}")

    hyp_path = Path(args.hyp)
    if hyp_path.is_dir():
        out = {}
        missing = []
        for rid in ids:
            p = hyp_path / f"{rid}.txt"
            if not p.exists():
                missing.append(str(p))
            else:
                out[rid] = p.read_text(encoding="utf-8")
        if missing:
            raise FileNotFoundError("missing hypothesis file(s):\n  " + "\n  ".join(missing))
        return out
    if hyp_path.is_file():
        text = hyp_path.read_text(encoding="utf-8")
        return {rid: text for rid in ids}
    raise FileNotFoundError(f"hypothesis path does not exist: {hyp_path}")


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
        prog="eval_asr",
        description="Word-level WER for Russian ASR hypotheses (RU normalization).",
        epilog=(
            "Example: python tools/eval_asr/eval_asr.py "
            "--manifest data/test/manifest.jsonl --hyp hyp/ "
            "--report reports/data_eval/asr_report"
        ),
    )
    parser.add_argument("--manifest", required=True, type=Path, help="manifest.jsonl")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hyp", type=Path,
                       help="directory with <id>.txt hypotheses, or a single .txt for all records")
    group.add_argument("--hyp-json", type=Path,
                       help="JSON mapping id -> hypothesis text (or list of {id,text})")
    parser.add_argument("--report", required=True, type=Path,
                        help="output base path; writes <base>.json and <base>.txt")
    args = parser.parse_args(argv)

    try:
        records = load_manifest(args.manifest)
        manifest_dir = args.manifest.resolve().parent
        hyps = load_hypotheses(args, records, manifest_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    per_record = []
    total_ref = total_hyp = total_s = total_d = total_i = 0
    for rec in records:
        rid = rec["id"]
        text_path = resolve_path(rec["text_path"], manifest_dir)
        if not text_path.exists():
            print(f"ERROR: reference text not found for {rid}: {text_path}", file=sys.stderr)
            return 2
        ref = words(text_path.read_text(encoding="utf-8"))
        hyp = words(hyps.get(rid, ""))
        s, d, ins = align_counts(ref, hyp)
        errors = s + d + ins
        wer = (errors / len(ref)) if ref else (0.0 if not hyp else 1.0)
        total_ref += len(ref)
        total_hyp += len(hyp)
        total_s += s
        total_d += d
        total_i += ins
        per_record.append({
            "id": rid,
            "duration_sec": rec.get("duration_sec"),
            "source": rec.get("source"),
            "license": rec.get("license"),
            "ref_words": len(ref),
            "hyp_words": len(hyp),
            "substitutions": s,
            "deletions": d,
            "insertions": ins,
            "errors": errors,
            "wer": round(wer, 6),
        })

    total_errors = total_s + total_d + total_i
    total_wer = (total_errors / total_ref) if total_ref else 0.0
    macro_wer = sum(r["wer"] for r in per_record) / len(per_record) if per_record else 0.0

    payload = {
        "tool": TOOL_NAME,
        "version": TOOL_VERSION,
        "manifest": str(args.manifest),
        "normalization": "lowercase; yo->ye; punctuation removed; whitespace collapsed",
        "summary": {
            "records": len(per_record),
            "ref_words": total_ref,
            "hyp_words": total_hyp,
            "substitutions": total_s,
            "deletions": total_d,
            "insertions": total_i,
            "errors": total_errors,
            "wer": round(total_wer, 6),
            "macro_wer": round(macro_wer, 6),
        },
        "records": per_record,
    }

    lines = [
        f"{TOOL_NAME} {TOOL_VERSION}",
        f"manifest: {args.manifest}",
        f"normalization: {payload['normalization']}",
        "",
        f"{'id':<24} {'dur_s':>8} {'ref_w':>6} {'hyp_w':>6} {'S':>4} {'D':>4} {'I':>4} {'err':>5} {'WER':>8}",
        "-" * 82,
    ]
    for r in per_record:
        dur = r["duration_sec"]
        dur_s = f"{dur:.2f}" if isinstance(dur, (int, float)) else "-"
        lines.append(
            f"{r['id']:<24} {dur_s:>8} {r['ref_words']:>6} {r['hyp_words']:>6} "
            f"{r['substitutions']:>4} {r['deletions']:>4} {r['insertions']:>4} "
            f"{r['errors']:>5} {r['wer']:>8.4f}"
        )
    lines += [
        "-" * 82,
        f"TOTAL ref_words={total_ref} hyp_words={total_hyp} "
        f"S={total_s} D={total_d} I={total_i} errors={total_errors}",
        f"WER (micro) = {total_wer:.6f}   WER (macro) = {macro_wer:.6f}",
        "",
        f"dependencies: python={sys.version.split()[0]} (no external runtime deps)",
    ]
    text = "\n".join(lines) + "\n"

    json_path, txt_path = write_report(args.report, payload, text)
    print(text)
    print(f"report_json={json_path}")
    print(f"report_txt={txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
