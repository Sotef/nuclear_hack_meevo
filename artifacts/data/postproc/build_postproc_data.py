#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build post-processing eval sets for the Russian offline transcription project.

Outputs (deterministic, seed=42):
  data/postproc/punct_eval.jsonl       >=300 punctuation restoration pairs
  data/postproc/correction_eval.jsonl  13 ASR->reference correction pairs
  data/postproc/build_stats.json       source/license/count statistics

Punctuation source: Tatoeba Russian sentences export
  https://downloads.tatoeba.org/exports/per_language/rus/rus_sentences.tsv.bz2
  License: CC-BY 2.0 FR (Tatoeba default for textual sentences; attribution required).
  Individual sentences may carry a different CC license; the export does not include
  per-sentence license metadata, so the conservative default CC-BY-2.0-FR is recorded.

Correction source: existing project artifacts only
  data/test/<id>.txt                              (reference, no punctuation)
  tools/eval_asr/hyp/small_zipformer_ru/<id>.txt  (ASR hypothesis, WER 0.146881)
"""
from __future__ import annotations

import bz2
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "postproc"
TATOEBA_BZ2 = Path(r"D:\temp\opencode\rus_sentences.tsv.bz2")
TATOEBA_URL = "https://downloads.tatoeba.org/exports/per_language/rus/rus_sentences.tsv.bz2"

SEED = 42
N_PUNCT = 300
N_CLEAN = 50

ALLOWED_RE = re.compile(r"^[А-Яа-яЁё\s.,!?;:—–\-…()\"'«»]+$")
WORD_RE = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")
PUNCT_RE = re.compile(r"[.,!?;:—–…]")

SOURCE_PUNCT = "Tatoeba Russian sentences (rus_sentences.tsv.bz2)"
LICENSE_PUNCT = "CC-BY-2.0-FR"


def ensure_tatoeba() -> Path:
    if TATOEBA_BZ2.exists() and TATOEBA_BZ2.stat().st_size > 0:
        return TATOEBA_BZ2
    import requests

    TATOEBA_BZ2.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {TATOEBA_URL} ...")
    r = requests.get(TATOEBA_URL, timeout=180)
    r.raise_for_status()
    TATOEBA_BZ2.write_bytes(r.content)
    return TATOEBA_BZ2


def normalize_ref(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def make_raw(text_ref: str) -> str:
    """Remove punctuation/markup, keep Cyrillic word tokens, join with single spaces."""
    return " ".join(WORD_RE.findall(text_ref))


def load_tatoeba(path: Path) -> list[dict]:
    rows: list[dict] = []
    seen_raw: set[str] = set()
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            sid, lang, text = parts
            if lang != "rus":
                continue
            text_ref = normalize_ref(text)
            if not text_ref or not ALLOWED_RE.match(text_ref):
                continue
            if not text_ref.endswith((".", "!", "?")):
                continue
            words = WORD_RE.findall(text_ref)
            if not (5 <= len(words) <= 30):
                continue
            if len(text_ref) > 300:
                continue
            # Require at least one internal punctuation mark or a non-period final mark,
            # so the pair actually exercises punctuation restoration.
            marks = PUNCT_RE.findall(text_ref)
            if len(marks) < 2:
                continue
            text_raw = make_raw(text_ref).lower()
            if len(text_raw.split()) != len(words):
                continue
            if text_raw in seen_raw:
                continue
            seen_raw.add(text_raw)
            rows.append({"tatoeba_id": int(sid), "text_raw": text_raw, "text_ref": text_ref})
    rows.sort(key=lambda r: r["tatoeba_id"])
    return rows


def build_punct(rows: list[dict]) -> list[dict]:
    if len(rows) < N_PUNCT:
        raise SystemExit(f"not enough Tatoeba candidates: {len(rows)} < {N_PUNCT}")
    rng = random.Random(SEED)
    picked = rng.sample(rows, N_PUNCT)
    picked.sort(key=lambda r: r["tatoeba_id"])
    out = []
    for i, r in enumerate(picked):
        out.append(
            {
                "id": f"punct_{i + 1:04d}",
                "text_raw": r["text_raw"],
                "text_ref": r["text_ref"],
                "source": SOURCE_PUNCT,
                "license": LICENSE_PUNCT,
                "tatoeba_id": r["tatoeba_id"],
                "clean": i < N_CLEAN,
            }
        )
    return out


def build_correction() -> list[dict]:
    ref_dir = ROOT / "data" / "test"
    hyp_dir = ROOT / "tools" / "eval_asr" / "hyp" / "small_zipformer_ru"
    out = []
    for ref_path in sorted(ref_dir.glob("*.txt")):
        rid = ref_path.stem
        hyp_path = hyp_dir / f"{rid}.txt"
        if not hyp_path.exists():
            raise SystemExit(f"missing ASR hypothesis for {rid}: {hyp_path}")
        text_ref = normalize_ref(ref_path.read_text(encoding="utf-8"))
        text_asr = normalize_ref(hyp_path.read_text(encoding="utf-8"))
        out.append(
            {
                "id": rid,
                "text_asr": text_asr,
                "text_ref": text_ref,
                "source": "data/test reference + tools/eval_asr/hyp/small_zipformer_ru",
                "license": "CC-BY-4.0 / MIT (see data/test/manifest.jsonl)",
            }
        )
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tatoeba = ensure_tatoeba()
    candidates = load_tatoeba(tatoeba)
    punct = build_punct(candidates)
    corr = build_correction()
    write_jsonl(OUT_DIR / "punct_eval.jsonl", punct)
    write_jsonl(OUT_DIR / "correction_eval.jsonl", corr)

    stats = {
        "seed": SEED,
        "tatoeba_candidates": len(candidates),
        "punct_pairs": len(punct),
        "punct_clean_marked": sum(1 for r in punct if r["clean"]),
        "punct_source": SOURCE_PUNCT,
        "punct_license": LICENSE_PUNCT,
        "correction_pairs": len(corr),
        "correction_source": "data/test + tools/eval_asr/hyp/small_zipformer_ru",
        "correction_license": "CC-BY-4.0 / MIT (per data/test/manifest.jsonl)",
    }
    (OUT_DIR / "build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
