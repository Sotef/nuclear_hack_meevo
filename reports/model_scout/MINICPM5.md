# MiniCPM5-2B — верификация как LLM-постпроцессора RU-транскрипции (Android arm64)

Дата: 2026-11-07. Автор: model-scout.
Статус: разведка по карточкам/докам (webfetch). Веса не скачивались, инференс/обучение не запускались.
Смежные отчёты: `reports/model_scout/LLM_CANDIDATES.md` (baseline QVikhr-3-1.7B, формат и обозначения —
не дублируются), `reports/asr_engineer/POSTPROC_REPORT.md` (punct F1 0.757, не пересчитывался).

Обозначения: **«не указано»** — нет в проверенном источнике; **«оценка»** — расчёт, не измерение.
Ревизия зафиксирована: `openbmb/MiniCPM5-2B` @ `f97400052a43d642bbc6e9975e2397e3ae6a6b52`
(lastModified 2026-09-29T06:12:33Z).

---

## 0. Краткий вывод

**MiniCPM5-2B существует** — это `openbmb/MiniCPM5-2B`, Apache-2.0, стандартный `LlamaForCausalLM`,
2.52B параметров, 131k контекста, релиз 06.09.2026. Ключевое отличие от QVikhr: **есть официальный
on-device артефакт LiteRT-LM (`.litertlm`, int4 1.55 ГБ) с замерами на Android-флаге** — это
разрешённый HARD CONSTRAINTS рантайм (LiteRT). Слабое место — **русский язык не заявлен**
(карточка: `language: en, zh`), RU-бенчмарков нет.

| Вопрос | Ответ |
|---|---|
| (а) постпроцесс без дообучения | **Низкая–средняя** (RU не заявлен, бенчмарков нет) |
| (б) дообучение на T-Wix | **Высокая** (Llama-архитектура, официальный TRL/PEFT-рецепт, влезает в 10 ГБ) |
| Итог | **Средняя**: технически лучший кандидат по пути «дообучение → LiteRT», но требует собственного RU-замера |

---

## 1. Идентификация

| Поле | Значение |
|---|---|
| Точное имя | `openbmb/MiniCPM5-2B` (instruct/post-trained версия — это и есть основной репозиторий; отдельного суффикса `-Instruct` нет) |
| URL | https://huggingface.co/openbmb/MiniCPM5-2B |
| Владелец | OpenBMB (ModelBest / BAAI-круг; GitHub `OpenBMB/MiniCPM`) |
| Ревизия | commit `f97400052a43d642bbc6e9975e2397e3ae6a6b52`; создан 2026-09-06, последнее изменение 2026-09-29 |
| Базовая архитектура | **`LlamaForCausalLM`**, `model_type: llama` — стандартный decoder-only, без custom code (`trust_remote_code=False`) |
| Параметры | **2 516 756 480** всего (2.52B), **1 981 982 720** без эмбеддингов. Название «2B» — маркетинговое; для расчётов памяти брать 2.52B |
| Слои / hidden | 42 слоя, hidden 2048, intermediate 6144, head_dim 128 |
| Attention | GQA: 16 Q-голов / **2 KV-головы** (агрессивное GQA → маленький KV-cache) |
| Контекст | **131 072** токена нативно (`max_position_embeddings`), `rope_theta = 5 000 000`, `rope_scaling: null` |
| Vocab | 130 560 (byte-level BPE), `tie_word_embeddings: false` (эмбеддинг и lm_head раздельные) |
| dtype | bf16 (`torch_dtype: bfloat16`), `transformers_version: 5.6.2` |
| Chat-шаблон | **ChatML**: `<\|im_start\|>role\n…<\|im_end\|>\n`; bos `<s>`, eos `</s>` (id 1) **и** `<\|im_end\|>` (id 130073). Есть блок ` thinking…</think>` и `enable_thinking`; при `enable_thinking=false` шаблон подставляет пустой ` thinking\n\n</think>\n\n`. Токены `/think` (130080) и `/no_think` (130081) присутствуют. Для постпроцессора — `enable_thinking=false` |
| Вариации семейства (реально существуют) | `MiniCPM5-2B` (final, RL+OPD), `-SFT`, `-Midtrain`, `-Base`, `-GGUF`, `-MLX`, `-GPTQ`, `-DSpark` (draft), `-DSpark-GGUF`, LiteRT `litert-community/MiniCPM5-2B`; младший `MiniCPM5-1B` (+`-Base`/`-SFT`/`-GGUF`/`-MLX`). Отдельного «MiniCPM5-2B-Instruction» нет |

---

## 2. Лицензия

| Поле | Значение |
|---|---|
| Лицензия | **Apache-2.0** — объявлена в метаданных карточки (`license: apache-2.0` в YAML-фронтматтере и в `cardData` HF API) |
| Ссылка | https://huggingface.co/openbmb/MiniCPM5-2B (блок лицензии в карточке); тег: https://huggingface.co/models?license=apache-2.0 |
| Файл LICENSE в репозитории | **Отсутствует** — в списке файлов репозитория (`siblings`) только README/README-cn, `chat_template.jinja`, `config.json`, `generation_config.json`, safetensors, index, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`. Юридически опора — тег/метаданные HF, не текст файла |
| Коммерческое использование | **Разрешено** (Apache-2.0) |
| Дообучение | **Разрешено**, включая закрытые производные; требуется сохранить уведомления и указать изменения |
| Требования | Сохранить copyright/NOTICE и текст Apache-2.0 при распространении; отметить модификации (для APK — экран «Лицензии»). Отдельного NOTICE-файла в репозитории нет |
| Производные артефакты | LiteRT-сборка `litert-community/MiniCPM5-2B` — тоже Apache-2.0, в карточке перечислены «Changes from the original work» (конвертация + квантизация, без fine-tune) |
| Данные обучения | Ultra-FineWeb / UltraX / UltraData-* — лицензии датасетов **не проверялись**; на лицензию Apache-2.0 весов не влияют, но зафиксировать в лицензионном аудите |

---

## 3. Веса и форматы

| Формат | Репозиторий | Размер | Годность для продукта |
|---|---|---|---|
| **safetensors BF16** | `openbmb/MiniCPM5-2B` | **5 033 557 096 Б ≈ 4.69 ГиБ** (один файл, sha256 `14fb8e7f…`) | Источник для конвертации/обучения |
| **LiteRT-LM (`.litertlm`) int4** | `litert-community/MiniCPM5-2B` | **1.55 ГБ** (blockwise-32 int4 + OCTAV на linear, int8 embedding) | **Да — разрешённый рантайм (LiteRT)** |
| **LiteRT-LM int8** | `litert-community/MiniCPM5-2B` | **2.55 ГБ** (`dynamic_wi8_afp32`) | Да, но тяжело для среднего телефона |
| GGUF | `openbmb/MiniCPM5-2B-GGUF` (+bartowski, turintech, axteam) | Q4_K_M ≈ 1.5–1.6 ГБ (оценка) | **Нет** — llama.cpp вне HARD CONSTRAINTS; только как источник весов |
| MLX 4bit | `openbmb/MiniCPM5-2B-MLX` | ≈1.4 ГБ (оценка) | Нет (Apple) |
| GPTQ 4bit | `openbmb/MiniCPM5-2B-GPTQ` | ≈1.4–1.5 ГБ (оценка) | Нет как рантайм; полезен как 4-bit чекпойнт для QLoRA |
| **ONNX** | официального **нет** | — | См. ниже |
| ONNX (community) | `RASMUS/MiniCPM5-2B-ONNX`, `Mike0021/…-ONNX` (int4), `tixiren/…-ONNX`, `nicolasembleton/…-onnx`, `anydaytv/minicpm5-2b-onnx-int4`, `liodon-ai/…-ONNX` (`license: other`) | не указано | Это **transformers.js / onnxruntime-web (WebGPU)** сборки, не пакеты ORT GenAI. Как есть для Android не годятся |

**ONNX-путь (оценка: высокая вероятность успеха).** Архитектура — `Llama`, а `Llama` **явно в списке
поддерживаемых** ONNX Runtime GenAI model builder (README `src/python/py/models/README.md`, раздел
«Current Support»). Значит официальный путь есть:

```
python -m onnxruntime_genai.models.builder -m openbmb/MiniCPM5-2B -o out -p int4 -e cpu
```

Плюсы: нет QK-norm/кастомных операторов (в отличие от Qwen3), GQA 16/2 и RoPE — стандартные.
Что проверить перед планированием: (1) vocab 130 560 и `rope_theta=5e6`; (2) 42 слоя и `head_dim=128`
(стандартные значения, но прогон builder на этой ревизии не выполнялся); (3) builder также умеет
LoRA-адаптеры (раздел «LoRA Models»), т.е. можно экспортировать base+adapter. Официального
ONNX-артефакта нет → это отдельная задача optimizer.

**LiteRT-путь (подтверждён замерами, не оценка).** Официальная конвертация — команда
`models/minicpm5` в `google-ai-edge/LiteRT-LM` (`litert-torch export_hf`, `--quantization_recipe=dynamic_wi8_afp32`).
Замеры из карточки LiteRT на **Galaxy S26 (SM-S942Q, Snapdragon SM8850, Adreno), Android 16**:

| Файл | Backend | Prefill (256) | Decode | Init | Peak RSS |
|---|---|---|---|---|---|
| int4 | GPU (OpenCL) | 525.1 tok/s | **24.4 tok/s** | 8.97 s (кэш) / 11.91 s | 1.14 ГБ |
| int4 | CPU | 100.7 tok/s | 19.7 tok/s | 5.05 / 6.00 s | 2.12 ГБ |
| int8 | GPU (OpenCL) | 542–549 tok/s | 22.4 tok/s | 5.6–6.4 s | 0.82 ГБ |
| int8 | CPU | 93–132 tok/s | 15.1–18.2 tok/s | 3.6–4.2 s | 3.07 ГБ |

Это флагман, не «средний класс» — на целевом телефоне ожидать ниже (оценка: 8–15 tok/s int4 GPU).
Для реплик в десятки токенов этого достаточно; TTFT 0.4–0.6 с.

> **Оговорка по констрейнту.** HARD CONSTRAINTS разрешают «TFLite/LiteRT». LiteRT-LM — это
> LLM-рантайм поверх LiteRT (преемник MediaPipe LLM Inference / TFLite LLM), т.е. формально
> попадает в пункт LiteRT. Но если пользователь трактует констрейнт узко («только TFLite-модели»),
> это **эскалация**: подтвердить, что LiteRT-LM допустим, до планирования интеграции.

**KV-cache (оценка):** 42 слоя × 2 KV-головы × 128 × 2 (K+V) × 2 байта = **42 КиБ/токен**;
2048 токенов ≈ 84 МиБ, 4096 ≈ 168 МиБ. Мало — сильная сторона GQA 16/2.

---

## 4. Дообучение (LoRA/QLoRA, RTX 3080 10 ГБ)

**Поддержка.** `transformers >= 5.6` (карточка: `pip install -U "transformers>=5.6"`; конфиг собран на 5.6.2),
`trust_remote_code=False`, `attn_implementation="sdpa"`. Официальные рецепты OpenBMB:
**TRL + PEFT** (`docs/finetune/trl.md`), **LLaMA-Factory**, **ms-swift**, **unsloth** — все перечислены в карточке.

**Официальный LoRA-конфиг (из `docs/finetune/trl.md`, не выдуман):**

```python
LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
SFTConfig(num_train_epochs=2, per_device_train_batch_size=4, gradient_accumulation_steps=4,
          learning_rate=2e-4, warmup_ratio=0.03, lr_scheduler_type="cosine", bf16=True,
          max_length=2048, packing=False, assistant_only_loss=True, seed=42)
```

Плюс training-only патч chat-шаблона с `{% generation %}` для assistant-only loss (в доке приведён целиком);
на инференсе грузить **оригинальный** tokenizer/шаблон.

**Оценка числа обучаемых параметров (расчёт):** r=16 на 7 модулях × 42 слоя ≈ **25.1M** (≈1.0% от 2.52B) —
совпадает с формулировкой доки «about 1%».

**Оценка памяти GPU (расчёт, не замер):**

| Режим | Веса базы | LoRA (fp32 вес+grad+Adam) | Активации (seq 2048, grad ckpt) | Итого (оценка) | Влезает в 10 ГБ? |
|---|---|---|---|---|---|
| LoRA BF16, batch 1 | 5.03 ГБ | ≈0.40 ГБ | ≈1.0–1.5 ГБ | **≈7.0–7.5 ГБ** | Да, с запасом ~2 ГБ |
| LoRA BF16, batch 4 (как в доке) | 5.03 ГБ | ≈0.40 ГБ | ≈3–4 ГБ | ≈9–10 ГБ | **На грани / вероятен OOM** → batch 1–2 + grad accum |
| QLoRA NF4 (bitsandbytes), batch 4 | ≈1.4 ГБ | ≈0.40 ГБ | ≈2–3 ГБ | **≈4.5–5.5 ГБ** | Да, комфортно |
| Full fine-tune BF16 + AdamW | 5.03 ГБ | — | — | ≈40 ГБ (16 байт/параметр) | **Нет** |

Практические выводы: (1) на 3080 10 ГБ рабочий режим — **LoRA bf16, seq 1024–2048, batch 1–2,
gradient_accumulation 8–16, gradient_checkpointing**; (2) QLoRA NF4 даёт запас и позволяет batch 4,
но **bitsandbytes на нативном Windows нестабилен** (проект на win32) → либо WSL2/Linux, либо LoRA bf16;
(3) официальный GPTQ-чекпойнт можно использовать как 4-bit инициализацию, но стандартный путь QLoRA —
bitsandbytes NF4, а не GPTQ+LoRA.

**Оценка скорости обучения (расчёт):** fwd+bwd ≈ 6·N·tokens ≈ 1.5·10¹⁰ FLOP/токен; при MFU ~20–25%
от ~60 TFLOPS FP16/BF16 tensor на 3080 → **≈800–1500 ток/с**. При 4096 ток/шаг (batch 2 × 2048) —
**≈3–5 с/шаг**, т.е. ~700–1200 шагов/час. Для T-Wix (размер набора в этом ревью не смотрелся) —
1 эпоха на 10k примеров × 512 токенов ≈ 1.5–2.5 ч (**оценка**).

**Риск сходимости:** assistant-only loss обязателен (`assistant_only_loss=True` + `{% generation %}`),
иначе loss по всем токенам и на малом адаптере может расти (прямо описано в Q&A доки).

---

## 5. Качество

**RU-бенчмарки: не указано.** В карточке `language: en, zh`; ни MERA, ни RuArena, ни RuBIN, ни
замеров пунктуации/исправления ASR-ошибок нет. Поиск по карточке (README + README-cn) слова
«Russian/multilingual/MERA» не даёт. Техрепорт, на который ссылается карточка (`arXiv:2602.09003`),
— это статья про tiered data management, а не RU-эвал; второй (`arXiv:2506.07900`) — MiniCPM4 tech report.

Опубликованное (карточка, сравнение с 2B/4B-классом; среднее **53.9**):

| Метрика | MiniCPM5-2B | Ближайший ориентир |
|---|---|---|
| MMLU-Pro | 70.8 | Qwen3.5-4B 78.0 |
| IFBench | 66.3 | granite-4.2-3B 73.0 |
| IFEval | 86.7 | LFM2.5-2.6B 93.4 |
| Multi-IF | 71.8 | LFM2.5-2.6B 76.8 |
| AIME 2025 | 86.5 | Qwen3.5-4B 78.8 |
| LiveCodeBench v6 | 69.1 | Qwen3.5-4B 56.4 |

**Сравнение с QVikhr-3-1.7B (одной строкой):** QVikhr имеет **подтверждённый русский**
(Ru Arena 59.2, MERA 36.8, RuBIN 33.0) и рабочий baseline пунктуации F1 0.757, а MiniCPM5-2B —
**сильнее в общих/инструктивных способностях** (IFBench 66.3, IFEval 86.7, 2.52B vs 1.7B),
но **RU-качество не измерено вообще**; сравнение на целевой задаче сейчас невозможно.

---

## 6. Вердикт

**(а) Постпроцесс без дообучения — низкая–средняя.** Модель не заявлена как русскоязычная, RU-эвала нет,
покрытие кириллицы токенизатором (130 560 BPE) не проверено — возможен раздутый расход токенов на
русском. Плюсы: сильный instruction-following и 131k контекст. Без собственного замера на
пунктуации/исправлении ASR-ошибок ставить её вместо QVikhr нельзя.

**(б) Дообучение на T-Wix — высокая.** Это самый «инженерно чистый» кандидат из рассмотренных:
стандартный `LlamaForCausalLM` без кастомного кода, официальный TRL/PEFT-рецепт с готовыми
`target_modules`, LoRA ≈25M параметров и ~7 ГБ пика на 3080 10 ГБ, официальный LiteRT int4-артефакт
(1.55 ГБ) с подтверждённым запуском на Android GPU/CPU, и `Llama` в списке архитектур ORT GenAI
model builder как альтернативный путь.

**Риски (по убыванию):**
1. **RU не заявлен и не измерен** — весь выигрыш зависит от T-Wix; нужен zero-shot замер до обучения.
2. **Токенизатор на кириллице** — не указано; влияет и на качество, и на скорость/размер.
3. **Нет официального ONNX** — только community web-сборки; ORT GenAI-путь требует проверки
   (42 слоя, GQA 16/2, vocab 130 560, rope_theta 5e6).
4. **LiteRT int4 теряет качество**: GSM8K 92% (bf16) → 86–87% (int4, thinking off); в thinking-режиме
   int4 на CPU не закрывает блок рассуждений (0/10). Для постпроцессора это приемлемо только с
   `enable_thinking=false`.
5. **Известный баг LiteRT-LM**: при thinking off со второго хода в KV-cache попадает stop-токен с нулевым
   эмбеддингом (файлы с отдельной embedder-секцией, к int4 относится). Обход — один запрос = одна
   «беседа» (для постпроцесса реплик это естественный режим) или int8-файл.
6. **Лицензия**: Apache-2.0 только тегом/метаданными, файла LICENSE в репозитории нет — зафиксировать
   URL и ревизию в лицензионном аудите; лицензии датасетов UltraData не проверены.
7. **Windows + bitsandbytes** для QLoRA — нестабильно; либо LoRA bf16, либо WSL2.

**Рекомендация:** держать MiniCPM5-2B как **кандидата №2 после QVikhr** — не заменять baseline, а
прогнать дешёвый гейт: zero-shot пунктуация/исправление на нашем RU-наборе (int4 LiteRT или bf16),
сравнить с F1 0.757. Если zero-shot приемлем или T-Wix-LoRA выводит его выше baseline — переходить
на него, потому что путь до Android (LiteRT int4 1.55 ГБ) у него короче и подтверждён замерами.

---

## 7. Источники (проверено 2026-11-07)

1. Карточка `openbmb/MiniCPM5-2B`: https://huggingface.co/openbmb/MiniCPM5-2B (Apache-2.0; 2 516 756 480 параметров; 42 слоя; GQA 16/2; 131 072; ChatML; список форматов и fine-tuning-фреймворков).
2. `config.json`: https://huggingface.co/openbmb/MiniCPM5-2B/raw/main/config.json (`LlamaForCausalLM`, hidden 2048, 42 слоя, vocab 130 560, rope_theta 5e6, bf16, transformers 5.6.2).
3. `tokenizer_config.json`: https://huggingface.co/openbmb/MiniCPM5-2B/raw/main/tokenizer_config.json (спецтокены `<|im_start|>`/`<|im_end|>`, `/think`, `/no_think`).
4. HF API (ревизия, размеры файлов, chat_template): https://huggingface.co/api/models/openbmb/MiniCPM5-2B?blobs=true (sha `f9740005…`; safetensors 5 033 557 096 Б).
5. LiteRT-LM сборка и замеры: https://huggingface.co/litert-community/MiniCPM5-2B (int4 1.55 ГБ / int8 2.55 ГБ; Galaxy S26: 24.4 tok/s GPU, 19.7 tok/s CPU; peak RSS 1.14/2.12 ГБ; известные баги).
6. Fine-tuning рецепт TRL+PEFT: https://github.com/OpenBMB/MiniCPM/blob/main/docs/finetune/trl.md (target_modules, r=16/alpha=32, lr 2e-4, assistant_only_loss).
7. ONNX Runtime GenAI model builder, поддерживаемые архитектуры (включая **Llama**) и LoRA-экспорт: https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/README.md
8. Поиск ONNX-сборок: https://huggingface.co/api/models?search=MiniCPM5-2B&filter=onnx (только community/transformers.js).
9. arXiv:2602.09003 — «Data Science and Technology Towards AGI Part I: Tiered Data Management» (09.02.2026), не RU-эвал: https://arxiv.org/abs/2602.09003

---

## 8. Открытые вопросы / что проверить перед следующим шагом

1. **Zero-shot RU-гейт**: пунктуация + исправление ASR-ошибок на нашем наборе, MiniCPM5-2B int4 (LiteRT)
   vs QVikhr int4 (F1 0.757). Без этого вердикт (а) не поднимается выше «средней».
2. **Токенизация русского**: среднее число токенов на слово/символ против Qwen3-токенизатора — влияет
   на RTF и на длину контекста в постпроцессе.
3. **Экспорт**: проверить `onnxruntime_genai.models.builder` на этой архитектуре (42 слоя, vocab 130 560,
   rope_theta 5e6) и сравнить int4 ORT vs int4 LiteRT по качеству/скорости.
4. **T-Wix LoRA**: 1 эпоха, seq 1024–2048, batch 1–2 + grad accum; замерить пик VRAM (ожидание ≈7 ГБ).
5. **Лицензионный аудит**: зафиксировать, что Apache-2.0 подтверждается только метаданными HF
   (файла LICENSE нет), и проверить лицензии UltraData/Ultra-FineWeb.
