#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merge ASR segments with diarization speakers by time overlap.

Version 0.2 adds optional LLM post-processing integration (additive fields
``text_raw``, ``doubtful``, ``edits`` in segments and ``llm_model``,
``postproc_version`` in meta). Without the new CLI options the tool keeps the
exact baseline-0.1 behaviour (backward compatible).

Algorithm (speakers)
--------------------
For every ASR segment [s, e] and every RTTM segment [rs, re] of speaker sp:
    overlap(sp) = sum over RTTM segments of sp of max(0, min(e, re) - max(s, rs))
The speaker with the largest total overlap wins. If the total overlap is zero
(no temporal intersection at all), the speaker of the nearest RTTM segment is
used (distance = gap between the two intervals). Ties are broken by speaker id
(lexicographic) and then by RTTM start, so the result is deterministic.

Algorithm (LLM text, only with --llm-hyp)
-----------------------------------------
The LLM is run per audio file, so its output is one string per record. To put
it back into ASR segments the tool aligns the record-level words to the
concatenated ASR segment words (difflib.SequenceMatcher on normalised words)
and assigns every LLM word to the segment of its aligned ASR word. The raw
text is split the same way from ``<id>.txt`` (the ASR full-text hypothesis).
Per-segment ``edits`` are a word-level diff ``text_raw -> text``; the reason is
taken from the LLM ``edits`` list when it matches, otherwise ``"punct"``.

Confidence is not produced by the ASR model, therefore it is written as 0.0
and documented as "not measured" in the report.

Usage
-----
    python merge.py --asr-hyp <dir> --rttm <dir> --manifest <path> --out <dir>
    python merge.py ... --force
    python merge.py ... --validate
    python merge.py --out <dir> --validate-only
    python merge.py ... --llm-hyp <jsonl> --llm-model <str> --postproc-version <str>
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

TOOL_VERSION = "1.1.0"
ASR_MODEL = "sherpa-onnx-small-zipformer-ru-2024-09-18 int8"
DIAR_MODEL = "campplus+pyannote-seg-3.0+FastClustering(thr=0.7)"
VERSION_V1 = "baseline-0.1"
VERSION_V2 = "baseline-0.2"
LANGUAGE = "ru"
SMALL_OVERLAP_SEC = 0.1
DURATION_TOLERANCE_SEC = 0.5
MIN_SEGMENT_SEC = 0.01  # repair for zero-length ASR segments (end == start)

TOP_KEYS = {"audio_id", "duration_sec", "language", "speakers", "segments", "meta"}
SPEAKER_KEYS = {"id", "label"}
SEGMENT_KEYS_V1 = {"speaker", "start", "end", "text", "confidence"}
SEGMENT_KEYS_V2 = SEGMENT_KEYS_V1 | {"text_raw", "doubtful", "edits"}
META_KEYS_V1 = {"asr_model", "diar_model", "version"}
META_KEYS_V2 = META_KEYS_V1 | {"llm_model", "postproc_version"}
EDIT_KEYS = {"from", "to", "reason"}

_PUNCT_CHARS = ".,!?;:—–…\"«»()[]{}'`"


class MergeError(Exception):
    """User-facing error with a clear message."""


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _norm_word(word: str) -> str:
    """Normalise a word for alignment: lowercase, ё->е, strip punctuation."""
    return word.lower().replace("ё", "е").strip(_PUNCT_CHARS)


def _norm_text(text: str) -> str:
    return " ".join(_norm_word(w) for w in text.split() if _norm_word(w))


def load_manifest(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise MergeError(f"manifest not found: {path}")
    records: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MergeError(f"manifest line {lineno}: invalid JSON ({exc})") from exc
            audio_id = obj.get("id")
            duration = obj.get("duration_sec")
            if not isinstance(audio_id, str) or not audio_id:
                raise MergeError(f"manifest line {lineno}: missing string 'id'")
            if not _is_number(duration) or duration <= 0:
                raise MergeError(f"manifest line {lineno}: invalid 'duration_sec' for {audio_id}")
            if audio_id in records:
                raise MergeError(f"manifest: duplicate id {audio_id!r}")
            records[audio_id] = float(duration)
    if not records:
        raise MergeError(f"manifest is empty: {path}")
    return records


def load_timestamps(path: Path) -> dict:
    if not path.is_file():
        raise MergeError(f"ASR timestamps not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise MergeError(f"ASR timestamps: invalid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise MergeError("ASR timestamps: top-level value must be an object keyed by audio id")
    return data


def load_llm_hyp(path: Path) -> dict[str, dict]:
    """Load a per-record LLM JSONL: {"id", "text", "doubtful"?, "edits"?}."""
    if not path.is_file():
        raise MergeError(f"LLM hypothesis file not found: {path}")
    records: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MergeError(f"{path.name} line {lineno}: invalid JSON ({exc})") from exc
            if not isinstance(obj, dict):
                raise MergeError(f"{path.name} line {lineno}: expected a JSON object")
            audio_id = obj.get("id")
            text = obj.get("text")
            if not isinstance(audio_id, str) or not audio_id:
                raise MergeError(f"{path.name} line {lineno}: missing string 'id'")
            if not isinstance(text, str):
                raise MergeError(f"{path.name} line {lineno}: missing string 'text' for {audio_id}")
            if audio_id in records:
                raise MergeError(f"{path.name}: duplicate id {audio_id!r}")
            doubtful = obj.get("doubtful") or []
            edits = obj.get("edits") or []
            if not isinstance(doubtful, list):
                raise MergeError(f"{path.name}: {audio_id}: 'doubtful' must be a list")
            if not isinstance(edits, list):
                raise MergeError(f"{path.name}: {audio_id}: 'edits' must be a list")
            records[audio_id] = {"text": text, "doubtful": doubtful, "edits": edits}
    if not records:
        raise MergeError(f"LLM hypothesis file is empty: {path}")
    return records


def parse_rttm(path: Path) -> list[dict]:
    if not path.is_file():
        raise MergeError(f"RTTM not found: {path}")
    segments: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                raise MergeError(f"{path.name} line {lineno}: expected SPEAKER RTTM record")
            try:
                start = float(parts[3])
                dur = float(parts[4])
            except ValueError as exc:
                raise MergeError(f"{path.name} line {lineno}: bad start/duration") from exc
            speaker = parts[7]
            if dur <= 0:
                continue
            segments.append({"start": start, "end": start + dur, "speaker": speaker})
    segments.sort(key=lambda s: (s["start"], s["end"], s["speaker"]))
    return segments


def assign_speaker(seg_start: float, seg_end: float, rttm: list[dict]) -> tuple[str, float, bool]:
    """Return (speaker, best_overlap_sec, used_nearest_fallback)."""
    if not rttm:
        raise MergeError("cannot assign speaker: RTTM has no SPEAKER segments")

    overlap_by_speaker: dict[str, float] = {}
    for rec in rttm:
        overlap = min(seg_end, rec["end"]) - max(seg_start, rec["start"])
        if overlap > 0:
            overlap_by_speaker[rec["speaker"]] = overlap_by_speaker.get(rec["speaker"], 0.0) + overlap

    if overlap_by_speaker:
        best_speaker = sorted(overlap_by_speaker.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        return best_speaker[0], best_speaker[1], False

    # No temporal intersection: nearest RTTM segment by interval gap.
    def gap(rec: dict) -> float:
        if seg_end < rec["start"]:
            return rec["start"] - seg_end
        if rec["end"] < seg_start:
            return seg_start - rec["end"]
        return 0.0

    nearest = sorted(rttm, key=lambda r: (gap(r), r["start"], r["speaker"]))[0]
    return nearest["speaker"], 0.0, True


def assign_words_to_segments(
    anchor_words: list[str], target_words: list[str], seg_word_counts: list[int]
) -> list[list[str]]:
    """Assign target words to ASR segments using anchor words as the timeline.

    ``anchor_words`` is the concatenation of the ASR segment words and
    ``seg_word_counts`` gives how many anchor words belong to each segment.
    ``target_words`` (raw full text or LLM text) is aligned to the anchor with
    difflib.SequenceMatcher on normalised words; every target word is mapped to
    the segment of its aligned anchor word. Insertions are attached to the
    nearest preceding anchor word (or the first one at the very beginning).
    """
    n_seg = len(seg_word_counts)
    if n_seg == 0:
        return []
    anchor_to_seg: list[int] = []
    for seg_idx, count in enumerate(seg_word_counts):
        anchor_to_seg.extend([seg_idx] * count)

    assigned: list[list[str]] = [[] for _ in range(n_seg)]
    if not target_words:
        return assigned
    if not anchor_words:
        assigned[0] = list(target_words)
        return assigned

    def seg_of_anchor(anchor_idx: int) -> int:
        if anchor_idx < 0:
            anchor_idx = 0
        if anchor_idx >= len(anchor_to_seg):
            anchor_idx = len(anchor_to_seg) - 1
        return anchor_to_seg[anchor_idx]

    a_norm = [_norm_word(w) for w in anchor_words]
    b_norm = [_norm_word(w) for w in target_words]
    matcher = difflib.SequenceMatcher(None, a_norm, b_norm, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(j2 - j1):
                assigned[seg_of_anchor(i1 + k)].append(target_words[j1 + k])
        elif tag == "replace":
            a_len = i2 - i1
            b_len = j2 - j1
            for k in range(b_len):
                if a_len > 0:
                    anchor_idx = i1 + min(a_len - 1, (k * a_len) // b_len)
                else:
                    anchor_idx = i1 - 1 if i1 > 0 else i1
                assigned[seg_of_anchor(anchor_idx)].append(target_words[j1 + k])
        elif tag == "insert":
            anchor_idx = i1 - 1 if i1 > 0 else i1
            for k in range(j2 - j1):
                assigned[seg_of_anchor(anchor_idx)].append(target_words[j1 + k])
        # "delete": target has no words for this anchor block
    return assigned


def _lookup_reason(raw_block: str, llm_block: str, llm_edits: list) -> str:
    """Use the LLM-provided reason when it matches, otherwise 'punct'."""
    if not llm_edits:
        return "punct"
    norm_from = _norm_text(raw_block)
    norm_to = _norm_text(llm_block)
    for edit in llm_edits:
        if not isinstance(edit, dict):
            continue
        edit_from = _norm_text(str(edit.get("from", "")))
        edit_to = _norm_text(str(edit.get("to", "")))
        if edit_from and edit_from == norm_from:
            if not edit_to or edit_to == norm_to:
                reason = edit.get("reason")
                return str(reason) if reason else "punct"
    return "punct"


def compute_edits(text_raw: str, text: str, llm_edits: list) -> list[dict]:
    """Word-level diff text_raw -> text with a reason per edit."""
    raw_tokens = text_raw.split()
    llm_tokens = text.split()
    matcher = difflib.SequenceMatcher(None, raw_tokens, llm_tokens, autojunk=False)
    edits: list[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        raw_block = " ".join(raw_tokens[i1:i2])
        llm_block = " ".join(llm_tokens[j1:j2])
        if not raw_block and not llm_block:
            continue
        edits.append(
            {
                "from": raw_block,
                "to": llm_block,
                "reason": _lookup_reason(raw_block, llm_block, llm_edits),
            }
        )
    return edits


def compute_doubtful(text: str, llm_doubtful: list) -> bool:
    """True if any LLM doubtful span occurs in this segment's text."""
    if not llm_doubtful:
        return False
    norm_text = _norm_text(text)
    for item in llm_doubtful:
        if isinstance(item, dict):
            span = item.get("span") or item.get("text") or ""
        else:
            span = item
        norm_span = _norm_text(str(span))
        if norm_span and norm_span in norm_text:
            return True
    return False


def build_document(
    audio_id: str,
    duration: float,
    asr_entry: dict,
    rttm: list[dict],
    llm_entry: dict | None = None,
    llm_model: str | None = None,
    postproc_version: str | None = None,
    raw_full_text: str | None = None,
) -> tuple[dict, dict]:
    if not isinstance(asr_entry, dict):
        raise MergeError(f"{audio_id}: ASR entry must be an object")
    raw_segments = asr_entry.get("segments")
    if not isinstance(raw_segments, list):
        raise MergeError(f"{audio_id}: ASR entry has no 'segments' list")

    speakers = sorted({rec["speaker"] for rec in rttm})
    if not speakers:
        raise MergeError(f"{audio_id}: RTTM has no speakers")

    # First pass: validate ASR segments and collect raw segment texts.
    parsed: list[dict] = []
    raw_segment_texts: list[str] = []
    for idx, seg in enumerate(raw_segments):
        if not isinstance(seg, dict):
            raise MergeError(f"{audio_id}: ASR segment #{idx} is not an object")
        try:
            start = float(seg["start"])
            end = float(seg["end"])
            text = seg["text"]
        except (KeyError, TypeError, ValueError) as exc:
            raise MergeError(f"{audio_id}: ASR segment #{idx} missing/invalid start/end/text") from exc
        if not isinstance(text, str):
            raise MergeError(f"{audio_id}: ASR segment #{idx} 'text' is not a string")
        if end < start:
            raise MergeError(f"{audio_id}: ASR segment #{idx} has end < start ({start}, {end})")
        parsed.append({"start": start, "end": end, "text": text})
        raw_segment_texts.append(text.strip())

    use_llm = llm_entry is not None
    if use_llm:
        if not llm_model or not postproc_version:
            raise MergeError(f"{audio_id}: --llm-model and --postproc-version are required with --llm-hyp")
        anchor_words: list[str] = []
        seg_word_counts: list[int] = []
        for seg_text in raw_segment_texts:
            words = seg_text.split()
            anchor_words.extend(words)
            seg_word_counts.append(len(words))
        raw_full_words = (raw_full_text if raw_full_text is not None else " ".join(raw_segment_texts)).split()
        llm_words = llm_entry["text"].split()
        raw_assigned = assign_words_to_segments(anchor_words, raw_full_words, seg_word_counts)
        llm_assigned = assign_words_to_segments(anchor_words, llm_words, seg_word_counts)
        llm_edits = llm_entry.get("edits") or []
        llm_doubtful = llm_entry.get("doubtful") or []

    out_segments: list[dict] = []
    stats = {
        "segments": 0,
        "no_overlap": 0,
        "small_overlap": 0,
        "repaired": 0,
        "segments_with_edits": 0,
        "edits": 0,
        "edit_chars": 0,
        "doubtful": 0,
    }
    for idx, seg in enumerate(parsed):
        start = seg["start"]
        end = seg["end"]
        if end == start:
            # ASR word end is the start time of the last token, so a one-word
            # segment can be zero-length. Extend minimally to satisfy start < end.
            end = start + MIN_SEGMENT_SEC
            stats["repaired"] += 1

        speaker, overlap, fallback = assign_speaker(start, end, rttm)
        stats["segments"] += 1
        if fallback:
            stats["no_overlap"] += 1
        elif overlap < SMALL_OVERLAP_SEC:
            stats["small_overlap"] += 1

        if use_llm:
            text_raw = " ".join(raw_assigned[idx]) or raw_segment_texts[idx]
            text = " ".join(llm_assigned[idx]) or text_raw
            edits = compute_edits(text_raw, text, llm_edits)
            doubtful = compute_doubtful(text, llm_doubtful)
            if edits:
                stats["segments_with_edits"] += 1
                stats["edits"] += len(edits)
                stats["edit_chars"] += sum(len(e["from"]) + len(e["to"]) for e in edits)
            if doubtful:
                stats["doubtful"] += 1
            out_segments.append(
                {
                    "speaker": speaker,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": text,
                    "confidence": 0.0,  # not measured: ASR model exposes no confidence
                    "text_raw": text_raw,
                    "doubtful": doubtful,
                    "edits": edits,
                }
            )
        else:
            out_segments.append(
                {
                    "speaker": speaker,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": seg["text"],
                    "confidence": 0.0,  # not measured: ASR model exposes no confidence
                }
            )

    out_segments.sort(key=lambda s: (s["start"], s["end"], s["speaker"]))

    meta = {"asr_model": ASR_MODEL, "diar_model": DIAR_MODEL, "version": VERSION_V1}
    if use_llm:
        meta = {
            "asr_model": ASR_MODEL,
            "diar_model": DIAR_MODEL,
            "version": VERSION_V2,
            "llm_model": llm_model,
            "postproc_version": postproc_version,
        }

    document = {
        "audio_id": audio_id,
        "duration_sec": round(float(duration), 3),
        "language": LANGUAGE,
        "speakers": [{"id": spk, "label": None} for spk in speakers],
        "segments": out_segments,
        "meta": meta,
    }
    return document, stats


def validate_document(doc: dict, expected_id: str | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(doc, dict):
        return ["document is not a JSON object"]
    if set(doc.keys()) != TOP_KEYS:
        errors.append(f"top-level keys {sorted(doc.keys())} != {sorted(TOP_KEYS)}")
        return errors

    if not isinstance(doc["audio_id"], str) or not doc["audio_id"]:
        errors.append("audio_id must be a non-empty string")
    elif expected_id is not None and doc["audio_id"] != expected_id:
        errors.append(f"audio_id {doc['audio_id']!r} != filename stem {expected_id!r}")

    if not _is_number(doc["duration_sec"]) or doc["duration_sec"] <= 0:
        errors.append("duration_sec must be a positive number")
    if doc["language"] != LANGUAGE:
        errors.append(f"language must be {LANGUAGE!r}")

    speakers = doc["speakers"]
    speaker_ids: list[str] = []
    if not isinstance(speakers, list) or not speakers:
        errors.append("speakers must be a non-empty list")
    else:
        for i, spk in enumerate(speakers):
            if not isinstance(spk, dict) or set(spk.keys()) != SPEAKER_KEYS:
                errors.append(f"speakers[{i}] must have exactly keys {sorted(SPEAKER_KEYS)}")
                continue
            if not isinstance(spk["id"], str) or not spk["id"]:
                errors.append(f"speakers[{i}].id must be a non-empty string")
            else:
                speaker_ids.append(spk["id"])
            if spk["label"] is not None:
                errors.append(f"speakers[{i}].label must be null")
        if len(set(speaker_ids)) != len(speaker_ids):
            errors.append("speakers contain duplicate ids")

    meta = doc["meta"]
    version = None
    if not isinstance(meta, dict):
        errors.append("meta must be an object")
        meta = {}
    else:
        version = meta.get("version")
    if version not in (VERSION_V1, VERSION_V2):
        errors.append(f"meta.version must be {VERSION_V1!r} or {VERSION_V2!r}, got {version!r}")
    is_v2 = version == VERSION_V2
    expected_meta = META_KEYS_V2 if is_v2 else META_KEYS_V1
    expected_segment = SEGMENT_KEYS_V2 if is_v2 else SEGMENT_KEYS_V1

    if isinstance(meta, dict):
        if set(meta.keys()) != expected_meta:
            errors.append(f"meta must have exactly keys {sorted(expected_meta)}")
        else:
            for key in expected_meta:
                if not isinstance(meta[key], str) or not meta[key]:
                    errors.append(f"meta.{key} must be a non-empty string")

    segments = doc["segments"]
    if not isinstance(segments, list):
        errors.append("segments must be a list")
        segments = []
    prev_start = None
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict) or set(seg.keys()) != expected_segment:
            errors.append(f"segments[{i}] must have exactly keys {sorted(expected_segment)}")
            continue
        if not isinstance(seg["speaker"], str) or seg["speaker"] not in speaker_ids:
            errors.append(f"segments[{i}].speaker {seg['speaker']!r} not in speakers")
        if not _is_number(seg["start"]) or not _is_number(seg["end"]):
            errors.append(f"segments[{i}] start/end must be numbers")
            continue
        if not seg["start"] < seg["end"]:
            errors.append(f"segments[{i}] start >= end")
        if seg["start"] < 0:
            errors.append(f"segments[{i}] start < 0")
        if prev_start is not None and seg["start"] < prev_start:
            errors.append(f"segments[{i}] not ordered by start")
        prev_start = seg["start"]
        if not isinstance(seg["text"], str):
            errors.append(f"segments[{i}].text must be a string")
        conf = seg["confidence"]
        if conf is not None and not _is_number(conf):
            errors.append(f"segments[{i}].confidence must be number or null")

        if is_v2:
            if not isinstance(seg["text_raw"], str):
                errors.append(f"segments[{i}].text_raw must be a string")
            if not isinstance(seg["doubtful"], bool):
                errors.append(f"segments[{i}].doubtful must be a bool")
            edits = seg["edits"]
            if not isinstance(edits, list):
                errors.append(f"segments[{i}].edits must be a list")
            else:
                for j, edit in enumerate(edits):
                    if not isinstance(edit, dict) or set(edit.keys()) != EDIT_KEYS:
                        errors.append(
                            f"segments[{i}].edits[{j}] must have exactly keys {sorted(EDIT_KEYS)}"
                        )
                        continue
                    for key in EDIT_KEYS:
                        if not isinstance(edit[key], str):
                            errors.append(f"segments[{i}].edits[{j}].{key} must be a string")

    return errors


def validate_directory(out_dir: Path) -> tuple[int, int, list[str]]:
    files = sorted(out_dir.glob("*.json"))
    if not files:
        raise MergeError(f"no *.json files to validate in {out_dir}")
    passed = 0
    messages: list[str] = []
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except json.JSONDecodeError as exc:
            messages.append(f"FAIL {path.name}: invalid JSON ({exc})")
            continue
        errors = validate_document(doc, expected_id=path.stem)
        if errors:
            messages.append(f"FAIL {path.name}: " + "; ".join(errors))
        else:
            passed += 1
            messages.append(f"PASS {path.name}")
    return passed, len(files), messages


def run_merge(args: argparse.Namespace) -> int:
    asr_dir = Path(args.asr_hyp)
    rttm_dir = Path(args.rttm)
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out)

    if not asr_dir.is_dir():
        raise MergeError(f"ASR hypothesis directory not found: {asr_dir}")
    if not rttm_dir.is_dir():
        raise MergeError(f"RTTM directory not found: {rttm_dir}")

    manifest = load_manifest(manifest_path)
    timestamps = load_timestamps(asr_dir / "timestamps.json")

    missing_ts = sorted(set(manifest) - set(timestamps))
    if missing_ts:
        raise MergeError(f"ASR timestamps missing ids: {', '.join(missing_ts)}")

    llm_map: dict[str, dict] | None = None
    if args.llm_hyp:
        llm_map = load_llm_hyp(Path(args.llm_hyp))
        missing_llm = sorted(set(manifest) - set(llm_map))
        if missing_llm:
            print(
                "[merge] WARNING: no LLM output for ids, falling back to text_raw: "
                + ", ".join(missing_llm),
                file=sys.stderr,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    targets = {audio_id: out_dir / f"{audio_id}.json" for audio_id in manifest}
    existing = [str(p) for p in targets.values() if p.exists()]
    if existing and not args.force:
        raise MergeError(
            "refusing to overwrite existing files without --force: " + ", ".join(existing)
        )

    total = {
        "segments": 0,
        "no_overlap": 0,
        "small_overlap": 0,
        "repaired": 0,
        "segments_with_edits": 0,
        "edits": 0,
        "edit_chars": 0,
        "doubtful": 0,
    }
    per_file: list[tuple[str, dict]] = []
    for audio_id in sorted(manifest):
        rttm = parse_rttm(rttm_dir / f"{audio_id}.rttm")
        llm_entry = llm_map.get(audio_id) if llm_map is not None else None
        raw_full_text = None
        if llm_entry is not None:
            txt_path = asr_dir / f"{audio_id}.txt"
            if txt_path.is_file():
                raw_full_text = txt_path.read_text(encoding="utf-8-sig").strip()
        doc, stats = build_document(
            audio_id,
            manifest[audio_id],
            timestamps[audio_id],
            rttm,
            llm_entry=llm_entry,
            llm_model=args.llm_model,
            postproc_version=args.postproc_version,
            raw_full_text=raw_full_text,
        )
        with targets[audio_id].open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        for key in total:
            total[key] += stats[key]
        per_file.append((audio_id, stats))
        print(
            f"[merge] {audio_id}: segments={stats['segments']} "
            f"no_overlap={stats['no_overlap']} small_overlap={stats['small_overlap']} "
            f"repaired={stats['repaired']} edits={stats['edits']} "
            f"doubtful={stats['doubtful']}"
        )

    print(
        f"[merge] total: files={len(per_file)} segments={total['segments']} "
        f"no_overlap={total['no_overlap']} small_overlap={total['small_overlap']} "
        f"repaired={total['repaired']} segments_with_edits={total['segments_with_edits']} "
        f"edits={total['edits']} edit_chars={total['edit_chars']} doubtful={total['doubtful']}"
    )

    if args.validate:
        passed, count, messages = validate_directory(out_dir)
        for message in messages:
            print(f"[validate] {message}")
        print(f"[validate] {passed}/{count} files valid")
        if passed != count:
            return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge ASR segments with diarization speakers by time overlap (OUTPUT CONTRACT).",
        epilog=(
            "Example: python merge.py --asr-hyp tools/eval_asr/hyp/small_zipformer_ru "
            "--rttm reports/diar_engineer/hyp/A_campplus_thr0.7 "
            "--manifest data/test/manifest.jsonl --out output/baseline --validate\n"
            "LLM example: python merge.py ... --out output/baseline_llm "
            "--llm-hyp reports/postproc_eval/llm_torch_correct13_punct.jsonl "
            "--llm-model \"Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning int4\" "
            "--postproc-version \"postproc_v1 (punctuate+correct)\" --validate"
        ),
    )
    parser.add_argument("--asr-hyp", help="directory with timestamps.json and <id>.txt")
    parser.add_argument("--rttm", help="directory with <id>.rttm diarization hypotheses")
    parser.add_argument("--manifest", help="path to manifest.jsonl")
    parser.add_argument("--out", required=True, help="output directory for <id>.json")
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    parser.add_argument("--validate", action="store_true", help="validate output after merging")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="only validate existing *.json in --out (no merge inputs required)",
    )
    parser.add_argument(
        "--llm-hyp",
        help="per-record LLM JSONL (id, text, doubtful?, edits?) to integrate (baseline-0.2)",
    )
    parser.add_argument("--llm-model", help="meta.llm_model value for baseline-0.2")
    parser.add_argument("--postproc-version", help="meta.postproc_version value for baseline-0.2")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.validate_only:
            passed, count, messages = validate_directory(Path(args.out))
            for message in messages:
                print(f"[validate] {message}")
            print(f"[validate] {passed}/{count} files valid")
            return 0 if passed == count else 1

        missing = [
            name
            for name in ("asr_hyp", "rttm", "manifest")
            if not getattr(args, name)
        ]
        if missing:
            parser.error("missing required arguments: " + ", ".join("--" + m.replace("_", "-") for m in missing))

        if args.llm_hyp:
            if not args.llm_model or not args.postproc_version:
                parser.error("--llm-hyp requires --llm-model and --postproc-version")
        elif args.llm_model or args.postproc_version:
            parser.error("--llm-model/--postproc-version require --llm-hyp")

        return run_merge(args)
    except MergeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
