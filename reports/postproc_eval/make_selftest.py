#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate deterministic self-test hypothesis files for eval_postproc."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "postproc_eval"
OUT.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


punct = read_jsonl(ROOT / "data" / "postproc" / "punct_eval.jsonl")
corr = read_jsonl(ROOT / "data" / "postproc" / "correction_eval.jsonl")

# 1) hypothesis == reference for punctuation; hypothesis == ASR for correction
ref_rows = []
for r in punct:
    ref_rows.append({"id": r["id"], "text": r["text_ref"]})
for r in corr:
    ref_rows.append({"id": r["id"], "text": r["text_asr"]})
write_jsonl(OUT / "selftest_ref.jsonl", ref_rows)

# 1b) hypothesis == reference for both punctuation and correction
corrected_rows = []
for r in punct:
    corrected_rows.append({"id": r["id"], "text": r["text_ref"]})
for r in corr:
    corrected_rows.append({"id": r["id"], "text": r["text_ref"]})
write_jsonl(OUT / "selftest_corrected.jsonl", corrected_rows)

# 2) same but commas removed from punctuation hypotheses
no_comma_rows = []
for r in punct:
    text = re.sub(r"\s*,\s*", " ", r["text_ref"])
    text = re.sub(r"\s+", " ", text).strip()
    no_comma_rows.append({"id": r["id"], "text": text})
for r in corr:
    no_comma_rows.append({"id": r["id"], "text": r["text_asr"]})
write_jsonl(OUT / "selftest_no_commas.jsonl", no_comma_rows)

# 3) flags: 5 clean sentences get a doubtful flag; correction pairs get a flag
#    on a word that is present in ASR but absent from the reference.
flag_rows = []
for i, r in enumerate(punct):
    text = r["text_ref"]
    if i < 5:
        first = re.search(r"[А-Яа-яЁё]+", text).group(0)
        text = text + f" [doubtful: {first} | вариант]"
    flag_rows.append({"id": r["id"], "text": text})
for r in corr:
    asr_words = re.findall(r"[А-Яа-яЁё]+", r["text_asr"].lower().replace("ё", "е"))
    ref_words = set(re.findall(r"[А-Яа-яЁё]+", r["text_ref"].lower().replace("ё", "е")))
    extra = next((w for w in asr_words if w not in ref_words), None)
    text = r["text_asr"]
    if extra:
        text = text + f" [doubtful: {extra}]"
    flag_rows.append({"id": r["id"], "text": text})
write_jsonl(OUT / "selftest_flags.jsonl", flag_rows)

print(
    "wrote selftest_ref.jsonl, selftest_corrected.jsonl, "
    "selftest_no_commas.jsonl, selftest_flags.jsonl"
)
