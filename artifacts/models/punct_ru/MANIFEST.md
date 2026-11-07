# RUPunct-small INT8 ONNX — манифест

Скачано: 2026-11-07. Автор: asr-engineer.

| Поле | Значение |
|---|---|
| Репозиторий | `ekhodzitsky/rupunct-small-onnx` |
| URL | https://huggingface.co/ekhodzitsky/rupunct-small-onnx |
| Ревизия (commit) | `c4a35be7e4a413cdfb08427db75b621b571569c9` (main) |
| Last modified | 2026-06-19 |
| Базовая модель | `RUPunct/RUPunct_small` (MIT) |
| Лицензия | **MIT** |
| Язык | ru |
| Архитектура | BERT token-classification (3 слоя, hidden 312, vocab 83828, 33 метки) |
| ONNX-файл | `rupunct_small_int8.onnx` |
| Размер ONNX | 29 259 673 байт (27.9 MiB) |
| SHA-256 ONNX | `B105DA023474D98AA13BA18953AE67B04B17BD0595034BC06030C17536893933` |
| Квантование | dynamic INT8 (DynamicQuantizeLinear + MatMulInteger) |

## Файлы

| Файл | Назначение |
|---|---|
| `rupunct_small_int8.onnx` | граф ONNX Runtime |
| `tokenizer.json` | HF WordPiece tokenizer (BertPreTokenizer) |
| `tokenizer_config.json` | do_lower_case=false, strip_accents=false |
| `config.json` | id2label/label2id (33 метки) |
| `special_tokens_map.json` | [PAD]/[UNK]/[CLS]/[SEP]/[MASK] |

## I/O

- Вход: `input_ids`, `attention_mask`, `token_type_ids` — int64 `[batch, seq]`.
- Выход: `logits` `[batch, seq, 33]` float32.
- Декод: aggregation `first` (метка первого субтокена слова) + `process_token`
  (3 режима регистра × 11 классов пунктуации). Слова соединяются одним пробелом.

## Воспроизведение

```powershell
.venv\Scripts\python.exe reports\asr_engineer\run_rupunct.py `
  --out reports\postproc_eval\rupunct_punct300.jsonl `
  --timing reports\postproc_eval\rupunct_punct300_timing.json
```
