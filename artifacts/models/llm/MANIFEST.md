# MANIFEST — LLM post-processor model (ONNX Runtime GenAI, int4)

Дата экспорта: 2026-11-06. Автор: asr-engineer.
Статус: ONNX int4-комплект собран локально и проверен на 10 примерах против
torch-версии (bf16, CUDA). Официального ONNX-артефакта у модели нет.

## 1. Исходная модель

| Поле | Значение |
|---|---|
| Имя | `Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning` |
| URL | https://huggingface.co/Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning |
| Ревизия (commit sha) | `ff7a23e896ebcd28825854b31cee34dfdaedd232` |
| Последнее изменение | 2025-05-30 15:39:18 UTC |
| Базовая модель | `Qwen/Qwen3-1.7B` (Apache-2.0) |
| Лицензия | Apache-2.0 (см. `LICENSE`, `NOTICE`) |
| Формат исходника | safetensors, BF16 |
| Размер исходника | `model.safetensors` = 3 441 185 608 байт (3.44 GB / 3.20 GiB) |
| sha256 исходника | `8c18861c5c1eb26003dead027e2a723c4fe63f1bf16406db6c5a4c2fc4941742` |
| Chat-шаблон | Qwen3 ChatML (`<|im_start|>role ... <|im_end|>`), `/no_think` |

## 2. Экспорт и квантизация

- Инструмент: ONNX Runtime GenAI model builder (`onnxruntime_genai.models.builder`).
- Архитектура: `qwen3`.
- Точность: **int4**, block size 32, accuracy level 4.
- Execution provider на этапе сборки: CPU (квантизация не зависит от GPU).
- Скрипт: `models/llm/export_qvikhr_int4.py`.
- Команда (эквивалент):

```powershell
.venv\Scripts\python.exe models\llm\export_qvikhr_int4.py
```

Скрипт вызывает:

```python
from onnxruntime_genai.models import builder
opts = builder.parse_extra_options(
    model_name="qwen3",
    input_path=r"C:\temp\opencode\qvikhr_src",
    output_dir=r"models\llm\QVikhr-3-1.7B-Instruction-noreasoning-int4",
    precision="int4",
    execution_provider="cpu",
    cache_dir=r"models\llm_build_cache",
    extra_options=["block_size=32", "accuracy_level=4"],
)
builder.create_model(
    model_name="qwen3",
    input_path=r"C:\temp\opencode\qvikhr_src",
    output_dir=r"models\llm\QVikhr-3-1.7B-Instruction-noreasoning-int4",
    precision="int4",
    execution_provider="cpu",
    cache_dir=r"models\llm_build_cache",
    **opts,
)
```

## 3. Артефакты int4-комплекта

Каталог: `models/llm/QVikhr-3-1.7B-Instruction-noreasoning-int4/`.

| Файл | Размер, байт | sha256 |
|---|---:|---|
| `model.onnx` | 290 594 | `f873cebe5159d4100da0063653a8d3fb3af563e5c5b83bcff1ec1a42d6f06ac0` |
| `model.onnx.data` | 1 096 749 056 | `2f74eeb412e745f00d88bffe108e4aee365825933fcd817ecfd3215e86c3e265` |
| `tokenizer.json` | 11 422 650 | `be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506` |
| `tokenizer_config.json` | 722 | `70dbff0c676db3313a55d21970d115b24a9bbc3f8bf7682c7d73cc948c7f7344` |
| `chat_template.jinja` | 4 214 | `3bd7d7e5b167e8b424759c8841ae488b2252167f7fddd6f8097fbea77c9b67c2` |
| `genai_config.json` | 1 668 | `160168836c9c6b4d981a586ae0a95d8a29a9d29e2df8631f7b3027e0f6478d42` |
| **Итого** | **1 108 468 904** (1.108 GB / 1.032 GiB) | — |

Сжатие относительно BF16: 3.44 GB → 1.108 GB, коэффициент ≈ **3.1×**
(теоретический предел int4 ≈ 4×; накладные расходы — embeddings/lm_head,
нормализации и метаданные).

`genai_config.json`: `type=qwen3`, 28 слоёв, hidden 2048, 16 heads, 8 KV heads,
context_length 40960, eos `[151645, 151643]`.

## 4. Версии окружения

| Компонент | Версия |
|---|---|
| Python | 3.12.10 |
| onnxruntime | 1.31.1 |
| onnxruntime-genai | 0.19.0 |
| onnx | 1.23.2 |
| torch (build-time, CUDA) | 2.11.0+cu128 |
| transformers | 5.22.2 |
| tokenizers | 0.23.3 |
| numpy | 2.5.4 |

## 5. Верификация ONNX int4 vs torch bf16

Набор: первые 10 предложений `data/postproc/punct_eval.jsonl`, режим
`punctuate`, greedy.

| Метрика | torch bf16 (CUDA) | ONNX int4 (CPU) |
|---|---:|---:|
| Точное совпадение текста | — | 3/10 |
| Средняя char-similarity (difflib) | — | **0.9856** |
| Punct micro F1 на 10 | 0.761905 | 0.681818 |
| Boundary F1 на 10 | 0.833333 | 0.720000 |
| Скорость | 1.08 с/предложение (300) | 1.51 с/предложение (10, CPU) |

Вывод: int4-комплект воспроизводит суть вывода torch-версии (среднее
посимвольное совпадение 98.6 %), но теряет часть пунктуации (F1 −0.08).
Комплект пригоден для Android-рантайма ONNX Runtime GenAI; для качества
пунктуации предпочтителен int8 или более крупная модель.

## 6. Ограничения

- Официального ONNX у QVikhr нет; комплект собран локально, не гарантируется
  поддержка сообществом.
- ONNX-бэкенд в текущем venv — CPU-only (`CPUExecutionProvider`); GPU-замеры
  сделаны на torch/CUDA, не на ONNX.
- int4 может деградировать на редких словах/именах; замер на 10 примерах —
  минимальная проверка, не полноценный quality gate.
- Лицензия датасета `Vikhrmodels/GrandMaster2` отдельно не проверялась;
  она не меняет Apache-2.0 лицензию весов, но должна быть зафиксирована в
  лицензионном аудите приложения.
