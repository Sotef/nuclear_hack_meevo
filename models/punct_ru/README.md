---
license: mit
language:
- ru
base_model: RUPunct/RUPunct_small
tags:
- onnx
- punctuation
- capitalization
- russian
- token-classification
library_name: onnxruntime
---

# RUPunct_small — INT8 ONNX

INT8-quantized ONNX export of [RUPunct/RUPunct_small](https://huggingface.co/RUPunct/RUPunct_small)
(MIT) — Russian **punctuation + capitalization restoration** as token classification.

Packaged for use in the [gigastt](https://github.com/ekhodzitsky/gigastt) speech-to-text
server: it runs via ONNX Runtime (no Python at runtime) with the Rust `tokenizers`
crate, restoring punctuation/casing on the plain `rnnt` head's bare lowercase output, e.g.
`шестьдесят тысяч тенге сколько будет стоить` → `Шестьдесят тысяч тенге, сколько будет стоить?`.

## Files
- `rupunct_small_int8.onnx` — dynamic-INT8 graph (DynamicQuantizeLinear + MatMulInteger), ~29 MB.
- `tokenizer.json`, `config.json`, `special_tokens_map.json`, `tokenizer_config.json` — HF WordPiece tokenizer + label map (33 labels).

## I/O
- Inputs: `input_ids`, `attention_mask`, `token_type_ids` (int64, `[batch, seq]`).
- Output: `logits` `[batch, seq, 33]`. Take the first sub-token label per word (aggregation `first`),
  then apply RUPunct's `process_token` decode (3 case modes × 11 punctuation classes).

## License
MIT, inherited from the upstream RUPunct/RUPunct_small. Export + INT8 quantization only;
weights are unchanged in accuracy (fp32 and int8 produce identical labels on test inputs).
