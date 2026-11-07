#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_postproc 1.0.0 - evaluate an LLM post-processor for Russian ASR text.

Metrics
-------
1. Punctuation restoration (vs data/postproc/punct_eval.jsonl):
   word-aligned precision / recall / F1 over punctuation marks
   (., !, ?, ,, ;, :, em/en dash, ellipsis) and sentence-boundary
   precision / recall / F1 (boundaries = . ! ? ...).
2. ASR correction (vs data/postproc/correction_eval.jsonl):
   word-level WER before (text_asr vs text_ref) and after
   (post-processed text vs text_ref), delta WER, and the share of
   improved / worsened / unchanged pairs.
3. Doubtful flags:
   - false-alarm proxy on the 50 clean sentences marked in punct_eval
     (clean=true): share of clean sentences that received >=1 doubtful flag;
   - coverage of known ASR errors: error lexemes are extracted from the
     word alignment of each correction pair (substitutions, deletions,
     insertions); a doubtful span or edit covers an error lexeme when the
     normalized surface form of the flag intersects the error lexeme set.

Hypothesis input format (--hyp)
-------------------------------
A JSONL file, one object per line:
  {"id": "punct_0001", "text": "Один, два, три.", "doubtful": [...], "edits": [...]}
  - "id"   : required, must match an id in --ref-punct or --ref-corr;
  - "text" : required, post-processed text (may contain inline markers);
  - "doubtful" (optional): list of strings or objects. A string is a doubtful
    span. An object may use "span", "text" or "from" for the span and
    "suggestions" for replacement variants.
  - "edits" (optional): list of objects {"from": "...", "to": "...", "reason": "..."}.
Alternatively --hyp may be a directory of <id>.txt files (inline markers only).

Inline doubtful markup inside "text" / .txt:
  [doubtful: слово]  or  [doubtful: слово | вариант1 | вариант2]
Markers are stripped before punctuation/WER scoring.

Exit codes: 0 = report written; 2 = bad input / refusing to overwrite.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

TOOL_VERSION = "1.0.0"

PUNCT_SET = set(".,!?;:—–…")
BOUNDARY_SET = set(".!?…")
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+(?:[-'][A-Za-zА-Яа-яЁё0-9]+)*")
TOKEN_RE = re.compile(
    r"[A-Za-zА-Яа-яЁё0-9]+(?:[-'][A-Za-zА-Яа-яЁё0-9]+)*|[.,!?;:—–…]"
)
DOUBTFUL_INLINE_RE = re.compile(r"\[doubtful:\s*(.*?)\]", re.IGNORECASE | re.DOTALL)
WER_STRIP_RE = re.compile(r"[^0-9a-zа-я\s]")


class EvalError(Exception):
    pass


def normalize_word(word: str) -> str:
    return word.lower().replace("ё", "е")


def strip_inline_doubtful(text: str) -> tuple[str, list[str]]:
    spans: list[str] = []

    def repl(match: re.Match) -> str:
        content = match.group(1)
        span = content.split("|", 1)[0].strip()
        if span:
            spans.append(span)
        return " "

    return DOUBTFUL_INLINE_RE.sub(repl, text), spans


def parse_words(text: str) -> list[tuple[str, frozenset[str]]]:
    """Return [(normalized_word, frozenset(marks_after_word)), ...]."""
    words: list[list] = []
    for match in TOKEN_RE.finditer(text):
        tok = match.group(0)
        if tok in PUNCT_SET:
            if words:
                words[-1][1].add(tok)
        else:
            words.append([normalize_word(tok), set()])
    return [(w, frozenset(p)) for w, p in words]


def normalize_for_wer(text: str) -> list[str]:
    text, _ = strip_inline_doubtful(text)
    text = text.lower().replace("ё", "е")
    text = WER_STRIP_RE.sub(" ", text)
    return text.split()


def wer_counts(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    """Levenshtein word errors and reference length."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m, 0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m], n


def safe_div(num: float, den: float, empty: float = 1.0) -> float:
    return num / den if den else empty


def f1_from_pr(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def score_punct(ref_text: str, hyp_text: str) -> dict:
    ref = parse_words(strip_inline_doubtful(ref_text)[0])
    hyp = parse_words(strip_inline_doubtful(hyp_text)[0])
    sm = SequenceMatcher(None, [w for w, _ in ref], [w for w, _ in hyp], autojunk=False)

    tp = fp = fn = 0
    ref_b = hyp_b = tp_b = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                r_marks = ref[i1 + k][1]
                h_marks = hyp[j1 + k][1]
                tp += len(r_marks & h_marks)
                fp += len(h_marks - r_marks)
                fn += len(r_marks - h_marks)
                rb = bool(r_marks & BOUNDARY_SET)
                hb = bool(h_marks & BOUNDARY_SET)
                ref_b += int(rb)
                hyp_b += int(hb)
                tp_b += int(rb and hb)
        else:
            for k in range(i1, i2):
                fn += len(ref[k][1])
                ref_b += int(bool(ref[k][1] & BOUNDARY_SET))
            for k in range(j1, j2):
                fp += len(hyp[k][1])
                hyp_b += int(bool(hyp[k][1] & BOUNDARY_SET))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    b_precision = safe_div(tp_b, hyp_b)
    b_recall = safe_div(tp_b, ref_b)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1_from_pr(precision, recall),
        "ref_boundaries": ref_b,
        "hyp_boundaries": hyp_b,
        "tp_boundaries": tp_b,
        "boundary_precision": b_precision,
        "boundary_recall": b_recall,
        "boundary_f1": f1_from_pr(b_precision, b_recall),
        "ref_words": len(ref),
        "hyp_words": len(hyp),
    }


def align_error_lexemes(asr_text: str, ref_text: str) -> set[str]:
    asr = normalize_for_wer(asr_text)
    ref = normalize_for_wer(ref_text)
    sm = SequenceMatcher(None, ref, asr, autojunk=False)
    errors: set[str] = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            errors.update(ref[i1:i2])
            errors.update(asr[j1:j2])
        elif tag == "delete":  # present in ref, missing in ASR
            errors.update(ref[i1:i2])
        elif tag == "insert":  # present in ASR, missing in ref
            errors.update(asr[j1:j2])
    return errors


def extract_flags(entry: dict) -> list[dict]:
    text = entry.get("text", "")
    _, inline_spans = strip_inline_doubtful(text)
    flags = [{"kind": "doubtful", "text": s, "suggestions": []} for s in inline_spans]

    doubtful = entry.get("doubtful") or []
    if isinstance(doubtful, (str, dict)):
        doubtful = [doubtful]
    for item in doubtful:
        if isinstance(item, str):
            span, suggestions = item, []
        elif isinstance(item, dict):
            span = item.get("span") or item.get("text") or item.get("from") or ""
            suggestions = item.get("suggestions") or []
        else:
            continue
        if span:
            flags.append({"kind": "doubtful", "text": str(span), "suggestions": suggestions})

    edits = entry.get("edits") or []
    if isinstance(edits, dict):
        edits = [edits]
    for item in edits:
        if isinstance(item, dict) and item.get("from"):
            flags.append(
                {
                    "kind": "edit",
                    "text": str(item["from"]),
                    "to": item.get("to"),
                    "reason": item.get("reason"),
                }
            )

    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for flag in flags:
        key = (flag["kind"], flag["text"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(flag)
    return unique


def flag_words(flag: dict) -> set[str]:
    return {normalize_word(w) for w in WORD_RE.findall(flag.get("text", ""))}


def load_jsonl(path: Path, required: tuple[str, ...]) -> list[dict]:
    if not path.exists():
        raise EvalError(f"reference file not found: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvalError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise EvalError(f"{path}:{lineno}: expected a JSON object")
            for key in required:
                if key not in obj:
                    raise EvalError(f"{path}:{lineno}: missing required field '{key}'")
                if not isinstance(obj[key], str):
                    raise EvalError(f"{path}:{lineno}: field '{key}' must be a string")
            rows.append(obj)
    if not rows:
        raise EvalError(f"reference file is empty: {path}")
    return rows


def load_hyp(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise EvalError(f"hypothesis path not found: {path}")
    entries: dict[str, dict] = {}
    if path.is_dir():
        for txt in sorted(path.glob("*.txt")):
            entries[txt.stem] = {"id": txt.stem, "text": txt.read_text(encoding="utf-8-sig")}
        if not entries:
            raise EvalError(f"no *.txt files in hypothesis directory: {path}")
        return entries

    with path.open("r", encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvalError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise EvalError(f"{path}:{lineno}: expected a JSON object")
            if "id" not in obj or "text" not in obj:
                raise EvalError(f"{path}:{lineno}: required fields are 'id' and 'text'")
            if not isinstance(obj["text"], str):
                raise EvalError(f"{path}:{lineno}: field 'text' must be a string")
            hid = str(obj["id"])
            if hid in entries:
                raise EvalError(f"{path}:{lineno}: duplicate id '{hid}'")
            entries[hid] = obj
    if not entries:
        raise EvalError(f"hypothesis file is empty: {path}")
    return entries


def aggregate_punct(per_sentence: list[dict]) -> dict:
    tp = sum(r["tp"] for r in per_sentence)
    fp = sum(r["fp"] for r in per_sentence)
    fn = sum(r["fn"] for r in per_sentence)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    ref_b = sum(r["ref_boundaries"] for r in per_sentence)
    hyp_b = sum(r["hyp_boundaries"] for r in per_sentence)
    tp_b = sum(r["tp_boundaries"] for r in per_sentence)
    b_precision = safe_div(tp_b, hyp_b)
    b_recall = safe_div(tp_b, ref_b)
    n = len(per_sentence)
    return {
        "sentences": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1_from_pr(precision, recall),
        "ref_boundaries": ref_b,
        "hyp_boundaries": hyp_b,
        "tp_boundaries": tp_b,
        "boundary_precision": b_precision,
        "boundary_recall": b_recall,
        "boundary_f1": f1_from_pr(b_precision, b_recall),
        "macro_precision": safe_div(sum(r["precision"] for r in per_sentence), n),
        "macro_recall": safe_div(sum(r["recall"] for r in per_sentence), n),
        "macro_f1": safe_div(sum(r["f1"] for r in per_sentence), n),
        "macro_boundary_f1": safe_div(sum(r["boundary_f1"] for r in per_sentence), n),
    }


def build_report(args: argparse.Namespace) -> dict:
    punct_ref = load_jsonl(Path(args.ref_punct), ("id", "text_raw", "text_ref"))
    corr_ref = load_jsonl(Path(args.ref_corr), ("id", "text_asr", "text_ref"))
    hyp = load_hyp(Path(args.hyp))

    punct_ids = {str(r["id"]) for r in punct_ref}
    corr_ids = {str(r["id"]) for r in corr_ref}
    known_ids = punct_ids | corr_ids
    unknown = sorted(set(hyp) - known_ids)

    datasets = {
        "punct": {
            "pairs": len(punct_ref),
            "clean_marked": sum(1 for r in punct_ref if r.get("clean") is True),
            "sources": sorted({str(r["source"]) for r in punct_ref if r.get("source")}),
            "licenses": sorted({str(r["license"]) for r in punct_ref if r.get("license")}),
        },
        "correction": {
            "pairs": len(corr_ref),
            "sources": sorted({str(r["source"]) for r in corr_ref if r.get("source")}),
            "licenses": sorted({str(r["license"]) for r in corr_ref if r.get("license")}),
        },
    }

    # (a) punctuation
    per_sentence: list[dict] = []
    punct_missing: list[str] = []
    for row in punct_ref:
        rid = str(row["id"])
        if rid not in hyp:
            punct_missing.append(rid)
            continue
        score = score_punct(row["text_ref"], hyp[rid]["text"])
        score["id"] = rid
        per_sentence.append(score)
    punct_summary = aggregate_punct(per_sentence) if per_sentence else None

    # (b) correction
    per_pair: list[dict] = []
    corr_missing: list[str] = []
    for row in corr_ref:
        rid = str(row["id"])
        if rid not in hyp:
            corr_missing.append(rid)
            continue
        ref_words = normalize_for_wer(row["text_ref"])
        asr_words = normalize_for_wer(row["text_asr"])
        hyp_words = normalize_for_wer(hyp[rid]["text"])
        before_err, n_ref = wer_counts(ref_words, asr_words)
        after_err, _ = wer_counts(ref_words, hyp_words)
        before = before_err / n_ref if n_ref else 0.0
        after = after_err / n_ref if n_ref else 0.0
        delta = after - before
        if delta < -1e-12:
            status = "improved"
        elif delta > 1e-12:
            status = "worsened"
        else:
            status = "unchanged"
        per_pair.append(
            {
                "id": rid,
                "ref_words": n_ref,
                "errors_before": before_err,
                "errors_after": after_err,
                "wer_before": before,
                "wer_after": after,
                "delta_wer": delta,
                "status": status,
            }
        )

    correction_summary = None
    if per_pair:
        total_ref = sum(p["ref_words"] for p in per_pair)
        before_micro = safe_div(sum(p["errors_before"] for p in per_pair), total_ref, 0.0)
        after_micro = safe_div(sum(p["errors_after"] for p in per_pair), total_ref, 0.0)
        n = len(per_pair)
        correction_summary = {
            "pairs": n,
            "ref_words": total_ref,
            "wer_before_micro": before_micro,
            "wer_after_micro": after_micro,
            "delta_wer_micro": after_micro - before_micro,
            "wer_before_macro": safe_div(sum(p["wer_before"] for p in per_pair), n, 0.0),
            "wer_after_macro": safe_div(sum(p["wer_after"] for p in per_pair), n, 0.0),
            "improved": sum(1 for p in per_pair if p["status"] == "improved"),
            "worsened": sum(1 for p in per_pair if p["status"] == "worsened"),
            "unchanged": sum(1 for p in per_pair if p["status"] == "unchanged"),
            "improved_share": safe_div(
                sum(1 for p in per_pair if p["status"] == "improved"), n, 0.0
            ),
            "worsened_share": safe_div(
                sum(1 for p in per_pair if p["status"] == "worsened"), n, 0.0
            ),
        }

    # (c) doubtful flags
    clean_rows = [r for r in punct_ref if r.get("clean") is True]
    clean_present = [r for r in clean_rows if str(r["id"]) in hyp]
    clean_with_flags = 0
    clean_flags = 0
    clean_words = 0
    for row in clean_present:
        entry = hyp[str(row["id"])]
        flags = [f for f in extract_flags(entry) if f["kind"] == "doubtful"]
        clean_flags += len(flags)
        clean_with_flags += int(bool(flags))
        clean_words += len(normalize_for_wer(entry["text"]))

    total_flags = sum(
        len([f for f in extract_flags(e) if f["kind"] == "doubtful"]) for e in hyp.values()
    )

    error_lexemes_total = 0
    error_lexemes_covered = 0
    pairs_with_errors = 0
    pairs_covered = 0
    for row in corr_ref:
        rid = str(row["id"])
        if rid not in hyp:
            continue
        errors = align_error_lexemes(row["text_asr"], row["text_ref"])
        if not errors:
            continue
        pairs_with_errors += 1
        error_lexemes_total += len(errors)
        covered: set[str] = set()
        for flag in extract_flags(hyp[rid]):
            covered |= errors & flag_words(flag)
        error_lexemes_covered += len(covered)
        pairs_covered += int(bool(covered))

    doubtful_summary = {
        "clean_sentences_total": len(clean_rows),
        "clean_sentences_evaluated": len(clean_present),
        "clean_sentences_with_flags": clean_with_flags,
        "false_alarm_rate": safe_div(clean_with_flags, len(clean_present), 0.0),
        "clean_flags": clean_flags,
        "clean_words": clean_words,
        "flag_density_per_100_words": (
            100.0 * clean_flags / clean_words if clean_words else 0.0
        ),
        "flags_total_all_hyp": total_flags,
        "error_lexemes_total": error_lexemes_total,
        "error_lexemes_covered": error_lexemes_covered,
        "error_coverage": safe_div(error_lexemes_covered, error_lexemes_total, 0.0),
        "pairs_with_errors": pairs_with_errors,
        "pairs_covered": pairs_covered,
        "pair_coverage": safe_div(pairs_covered, pairs_with_errors, 0.0),
    }

    return {
        "tool": "eval_postproc",
        "version": TOOL_VERSION,
        "dependencies": {
            "python": platform.python_version(),
            "third_party": [],
            "note": "stdlib only (json, re, difflib, argparse, pathlib)",
        },
        "datasets": datasets,
        "inputs": {
            "hyp": str(args.hyp),
            "hyp_entries": len(hyp),
            "ref_punct": str(args.ref_punct),
            "ref_punct_entries": len(punct_ref),
            "ref_corr": str(args.ref_corr),
            "ref_corr_entries": len(corr_ref),
            "unknown_hyp_ids": unknown[:20],
            "unknown_hyp_ids_count": len(unknown),
            "punct_missing_ids": punct_missing[:20],
            "punct_missing_count": len(punct_missing),
            "corr_missing_ids": corr_missing[:20],
            "corr_missing_count": len(corr_missing),
        },
        "punctuation": {
            "summary": punct_summary,
            "per_sentence": per_sentence,
        },
        "correction": {
            "summary": correction_summary,
            "per_pair": per_pair,
        },
        "doubtful": doubtful_summary,
    }


def fmt(value, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_text(report: dict) -> str:
    lines: list[str] = []
    lines.append(f"eval_postproc {report['version']}")
    lines.append(f"Python {report['dependencies']['python']}")
    inp = report["inputs"]
    lines.append("")
    lines.append("Inputs")
    lines.append(f"  hyp:        {inp['hyp']} ({inp['hyp_entries']} entries)")
    lines.append(f"  ref-punct:  {inp['ref_punct']} ({inp['ref_punct_entries']} entries)")
    lines.append(f"  ref-corr:   {inp['ref_corr']} ({inp['ref_corr_entries']} entries)")
    if inp["unknown_hyp_ids_count"]:
        lines.append(f"  unknown hyp ids: {inp['unknown_hyp_ids_count']}")
    if inp["punct_missing_count"]:
        lines.append(f"  punct ids missing in hyp: {inp['punct_missing_count']}")
    if inp["corr_missing_count"]:
        lines.append(f"  correction ids missing in hyp: {inp['corr_missing_count']}")

    ds = report.get("datasets", {})
    if ds:
        lines.append("")
        lines.append("Datasets")
        for name in ("punct", "correction"):
            d = ds.get(name)
            if not d:
                continue
            lines.append(
                f"  {name}: pairs={d['pairs']} "
                f"sources={'; '.join(d['sources'])} "
                f"licenses={'; '.join(d['licenses'])}"
            )

    lines.append("")
    lines.append("Punctuation (word-aligned marks)")
    p = report["punctuation"]["summary"]
    if p is None:
        lines.append("  not measured: no punctuation ids present in hyp")
    else:
        lines.append(
            f"  micro: P={fmt(p['precision'])} R={fmt(p['recall'])} F1={fmt(p['f1'])} "
            f"(TP={p['tp']} FP={p['fp']} FN={p['fn']}, sentences={p['sentences']})"
        )
        lines.append(
            f"  macro: P={fmt(p['macro_precision'])} R={fmt(p['macro_recall'])} "
            f"F1={fmt(p['macro_f1'])}"
        )
        lines.append(
            f"  sentence boundaries: P={fmt(p['boundary_precision'])} "
            f"R={fmt(p['boundary_recall'])} F1={fmt(p['boundary_f1'])} "
            f"(TP={p['tp_boundaries']} ref={p['ref_boundaries']} hyp={p['hyp_boundaries']})"
        )

    lines.append("")
    lines.append("Correction (WER vs text_ref)")
    c = report["correction"]["summary"]
    if c is None:
        lines.append("  not measured: no correction ids present in hyp")
    else:
        lines.append(
            f"  pairs={c['pairs']} ref_words={c['ref_words']} "
            f"WER before={fmt(c['wer_before_micro'])} after={fmt(c['wer_after_micro'])} "
            f"delta={fmt(c['delta_wer_micro'])}"
        )
        lines.append(
            f"  improved={c['improved']} ({fmt(c['improved_share'])}) "
            f"worsened={c['worsened']} ({fmt(c['worsened_share'])}) "
            f"unchanged={c['unchanged']}"
        )
        lines.append(
            f"  macro WER before={fmt(c['wer_before_macro'])} "
            f"after={fmt(c['wer_after_macro'])}"
        )

    lines.append("")
    lines.append("Doubtful flags")
    d = report["doubtful"]
    lines.append(
        f"  clean sentences: {d['clean_sentences_evaluated']}/{d['clean_sentences_total']} "
        f"evaluated, with flags={d['clean_sentences_with_flags']}, "
        f"false-alarm rate={fmt(d['false_alarm_rate'])}"
    )
    lines.append(
        f"  flags on clean text={d['clean_flags']} / {d['clean_words']} words "
        f"({fmt(d['flag_density_per_100_words'], 3)} per 100 words); "
        f"flags total in hyp={d['flags_total_all_hyp']}"
    )
    lines.append(
        f"  known ASR error lexemes: {d['error_lexemes_covered']}/{d['error_lexemes_total']} "
        f"covered (coverage={fmt(d['error_coverage'])}); "
        f"pairs covered={d['pairs_covered']}/{d['pairs_with_errors']} "
        f"({fmt(d['pair_coverage'])})"
    )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an LLM post-processor for Russian ASR: punctuation restoration, "
            "WER correction delta, and doubtful-flag quality."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Hypothesis JSONL example:\n"
            '  {"id":"punct_0001","text":"Один, два, три.","doubtful":["три"],'
            '"edits":[{"from":"три","to":"два","reason":"ASR"}]}\n'
            "  doubtful: list of strings or objects (span/text/from, suggestions);\n"
            "  edits: list of {from, to, reason}.\n"
            "Inline marker example:\n"
            '  {"id":"punct_0002","text":"Даже не знаю [doubtful: создан | создан ли], ..."}\n'
            "  --hyp may also be a directory of <id>.txt files (inline markers only).\n\n"
            "Example:\n"
            "  .venv\\Scripts\\python.exe tools\\eval_postproc\\eval_postproc.py "
            "--hyp hyp.jsonl --ref-punct data\\postproc\\punct_eval.jsonl "
            "--ref-corr data\\postproc\\correction_eval.jsonl "
            "--report reports\\postproc_eval\\report.json"
        ),
    )
    parser.add_argument("--hyp", required=True, help="JSONL file or directory of <id>.txt")
    parser.add_argument("--ref-punct", required=True, help="punct_eval.jsonl")
    parser.add_argument("--ref-corr", required=True, help="correction_eval.jsonl")
    parser.add_argument(
        "--report",
        required=True,
        help="report path; .json is written and a sibling .txt is derived",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing report files"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(args)
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if report["punctuation"]["summary"] is None and report["correction"]["summary"] is None:
        print(
            "error: no hypothesis ids matched --ref-punct or --ref-corr; "
            "check the 'id' fields",
            file=sys.stderr,
        )
        return 2

    report_path = Path(args.report)
    if report_path.suffix.lower() == ".json":
        json_path = report_path
    else:
        json_path = report_path.with_suffix(".json")
    txt_path = json_path.with_suffix(".txt")

    existing = [p for p in (json_path, txt_path) if p.exists()]
    if existing and not args.force:
        print(
            "error: report file(s) already exist; refusing to overwrite without --force: "
            + ", ".join(str(p) for p in existing),
            file=sys.stderr,
        )
        return 2

    json_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    text = render_text(report)
    txt_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"report written: {json_path}")
    print(f"report written: {txt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
