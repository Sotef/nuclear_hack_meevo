"""RUPunct-small ONNX runner for punct_eval.

Reproduces RUPunct's official decode:
  tokenizer: BertTokenizer, strip_accents=False, add_prefix_space=True
  aggregation: "first" (first sub-token label per word)
  process_token: 3 case modes x 11 punctuation classes
Output words joined by a single space (as in upstream README).
"""
import argparse
import json
import sys
import time

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_DIR = "models/punct_ru"
ONNX_PATH = MODEL_DIR + "/rupunct_small_int8.onnx"

PUNCT = {
    "O": "",
    "PERIOD": ".",
    "COMMA": ",",
    "QUESTION": "?",
    "TIRE": "\u2014",
    "DVOETOCHIE": ":",
    "VOSKL": "!",
    "PERIODCOMMA": ";",
    "DEFIS": "-",
    "MNOGOTOCHIE": "...",
    "QUESTIONVOSKL": "?!",
}


def process_token(token, label):
    # label like UPPER_TOTAL_PERIOD / LOWER_COMMA / UPPER_O
    if label.startswith("UPPER_TOTAL_"):
        case, cls = "upper", label[len("UPPER_TOTAL_"):]
    elif label.startswith("UPPER_"):
        case, cls = "cap", label[len("UPPER_"):]
    elif label.startswith("LOWER_"):
        case, cls = "lower", label[len("LOWER_"):]
    else:
        raise ValueError("bad label " + label)
    if case == "upper":
        word = token.upper()
    elif case == "cap":
        word = token.capitalize()
    else:
        word = token
    p = PUNCT[cls]
    if p == "\u2014" and case != "lower":
        return word + " " + p
    return word + p


def load():
    tok = AutoTokenizer.from_pretrained(
        MODEL_DIR, strip_accents=False, add_prefix_space=True
    )
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    return tok, sess


def punctuate(text, tok, sess):
    enc = tok(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )
    input_ids = enc["input_ids"].astype(np.int64)
    attention_mask = enc["attention_mask"].astype(np.int64)
    token_type_ids = enc.get("token_type_ids")
    if token_type_ids is None:
        token_type_ids = np.zeros_like(input_ids)
    token_type_ids = token_type_ids.astype(np.int64)
    offsets = enc["offset_mapping"][0]
    word_ids = enc.word_ids(0)

    logits = sess.run(
        ["logits"],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )[0][0]
    pred_ids = logits.argmax(axis=-1)
    id2label = sess.get_modelmeta().custom_metadata_map.get("id2label")
    # fall back to config labels
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    id2label = cfg.id2label

    # aggregation "first": first sub-token label per word, full word span
    # group subtoken indices by word id
    groups = {}
    order = []
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid not in groups:
            groups[wid] = []
            order.append(wid)
        groups[wid].append(i)
    words = []
    for wid in order:
        idxs = groups[wid]
        start = int(offsets[idxs[0]][0])
        end = int(offsets[idxs[-1]][1])
        if start == end:
            continue
        word = text[start:end]
        label = id2label[int(pred_ids[idxs[0]])]
        words.append(process_token(word, label))
    return " ".join(words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/postproc/punct_eval.jsonl")
    ap.add_argument("--out", default="reports/postproc_eval/rupunct_punct300.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timing", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    tok, sess = load()
    out = []
    t0 = time.perf_counter()
    for r in rows:
        hyp = punctuate(r["text_raw"], tok, sess)
        out.append({"id": r["id"], "text": hyp})
    dt = time.perf_counter() - t0

    with open(args.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    n = len(rows)
    print(f"segments={n} total_s={dt:.3f} sec_per_seg={dt/n:.4f}")
    if args.timing:
        with open(args.timing, "w", encoding="utf-8") as f:
            json.dump(
                {"segments": n, "total_s": dt, "sec_per_seg": dt / n}, f
            )
    for o in out[:5]:
        print(o["id"], "|", o["text"])


if __name__ == "__main__":
    main()
