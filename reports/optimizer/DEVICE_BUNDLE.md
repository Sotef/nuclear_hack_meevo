# DEVICE_BUNDLE — статус on-device пайплайна (SM-N9750, Android 12, arm64-v8a)

Дата: 2026-11-07. Агент: optimizer (повтор после падения по диску).

## Итог по уровням

| Уровень | Статус | Причина / цифры |
|---|---|---|
| ASR (sherpa-onnx small-zipformer-ru int8) | **не измерено на устройстве** | Модели и `.so` собраны в bundle; CLI-бинарь `sherpa-onnx-offline` в `android.tar.bz2` отсутствует (только `jniLibs`). Бинарь есть в `android-aarch64-termux-shared` (пере-скачиваем). Push/прогон не выполнен. |
| Диаризация (pyannote seg + 3dspeaker CAM++) | **не измерено на устройстве** | В релизе sherpa-onnx v2.0.1 есть `sherpa-onnx-offline-speaker-diarization` (termux-shared), но бинарь не включён в bundle (удалён при очистке диска). Кастомный C++ не строился. |
| LLM (MiniCPM5-2B+LoRA smoke, ORT GenAI int4) | **не измерено на устройстве** | Экспорт int4 **успешен** (1.64 ГБ), но ORT GenAI 0.19.0 не принимает `tokenizer.json` MiniCPM5: regex `\p{L}`/`\p{N}` не поддерживается; после ASCII/Cyrillic-замены — `invalid string_view position`. LiteRT-фолбэк не запускался. |

## Экспорт int4 (готово)

- Команда: `python tools/export_llm/export_llm.py --base C:\temp\opencode\minicpm5_base --lora models\llm_minicpm5\smoke --out models\llm_minicpm5\smoke_int4 --merged-in models\llm_minicpm5\smoke_merged --format ort-genai-int4 --verify 10 --device cuda --force`
- Результат: `models/llm_minicpm5/smoke_int4/` — `model.onnx` (300 488 B), `model.onnx.data` (1 640 669 184 B), `genai_config.json` (1 635 B), `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`.
- Время экспорта: 76.7 с (после загрузки merged-модели). Лог: `logs/export_smoke_int4.log`, `logs/export_smoke_int4.err`.
- Формат: ORT GenAI int4, block 32, accuracy 4; embedding/LM head — int4 (по логу `weight_Q4`).

## Паритет torch vs int4

- Torch-прогон (smoke_merged, CUDA bf16, greedy, 128 токенов): 10/10 примеров из `data/postproc/punct_eval.jsonl`, 70.73 с (7.074 с/пример). Выход: `logs/parity_torch10.jsonl`.
- ONNX int4-прогон: **не выполнен** — блокер токенизатора (см. выше). `logs/parity_onnx10.jsonl` отсутствует.
- Char-similarity / exact matches: **не посчитаны** (нет onnx-выхода). Это ограничение, а не «0 совпадений».

## Bundle

- `device_bundle/` — 1.62 ГБ: `models/asr` (int8 zipformer, 27 МБ), `models/diar` (3dspeaker 28 МБ + pyannote seg), `models/llm` (int4, 1.64 ГБ), `bin/sherpa` (`.so` arm64-v8a), `audio/` (ruls_0001, diar_0002, diar_0003), `scripts/push_to_device.ps1`, `scripts/run_stage.ps1`, `MANIFEST.md`.
- SHA256 ключевых файлов — в `device_bundle/MANIFEST.md`.

## Диск

- Старт: D: ~7.2 ГБ свободно.
- Освобождено (с записью в journal): `D:\hf_cache\models--ai-forever--ru-en-RoSBERTa` (1.5 ГБ), `D:\hf_cache\blobs` (3.78 ГБ, HF-кэш, пере-скачиваем), `data/twix/train_smoke.jsonl` (72 МБ, пере-скачиваем), скачанные архивы.
- После bundle: D: 8.78 ГБ свободно.
- Причина прошлых падений экспорта: builder писал промежуточный bf16-файл ~5.7–6.8 ГБ + int4-выход ~1.6 ГБ; при 8.2 ГБ свободного места — ENOSPC. После освобождения ~12 ГБ экспорт прошёл.

## Ограничения / следующие шаги

1. Починить токенизатор под ORT GenAI: заменить `pre_tokenizer` regex на ECMAScript-совместимый (без `\p{L}`/`\p{N}`) и проверить `encode`; либо собрать `tokenizer.json` из `tokenizer.model`/`vocab` в формате, который принимает ORT GenAI.
2. Скачать `sherpa-onnx-v2.0.1-android-aarch64-termux-shared.tar.bz2`, положить `bin/sherpa-onnx-offline` и `bin/sherpa-onnx-offline-speaker-diarization` в `device_bundle/bin/sherpa/`, запустить `push_to_device.ps1` и `run_stage.ps1`.
3. Собрать `llm_cli.cpp` NDK r27c (`aarch64-linux-android24`) с `libonnxruntime-genai.so` + `libmat.so` из AAR 0.19.0; после фикса токенизатора — smoke 16 токенов greedy.
4. LiteRT-фолбэк: `pip install litert-torch` и `export_llm.py --format litert-int4` (не запускалось).
