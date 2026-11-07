# Исходный код

## Структура

- `mobile/android/` — приложение Meevo (Kotlin). Сборка и запуск — ниже.
- `tools/` — python-инструменты исследований и eval (WER/DER/пунктуация,
  LoRA-обучение, подготовка данных). Зависимости — `tools/requirements.txt`,
  окружение — `.venv` в корне репозитория.

## Инструкция по сборке (Android)

Требования: JDK 17, Android SDK 34 (build-tools 34.0.0); см.
[docs/SYSTEM_REQUIREMENTS.md](../../docs/SYSTEM_REQUIREMENTS.md).

```bash
cd mobile/android
gradle assembleDebug      # результат: app/build/outputs/apk/debug/app-debug.apk
```

Модели (GigaAM, диаризация, RUPunct, QVikhr) находятся в
`app/src/main/assets/models/` и **не входят в git** — восстановите их по
`artifacts/models/*/MANIFEST.md` (URL + sha256) и скопируйте в assets перед
сборкой. Структура assets:

```
app/src/main/assets/
  models/asr/    — GigaAM v3: encoder.int8.onnx, decoder.onnx, joiner.onnx, tokens.txt
  models/diar/   — model.fixed.onnx, 3dspeaker campplus
  models/punct/  — rupunct_small_int8.onnx, tokenizer.json, config.json
  models/llm/    — QVikhr int4 (genai_config.json, model.onnx, model.onnx.data, tokenizer)
  prompts/       — system_prompt_ru.txt
```

Установка на устройство:

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## Инструкция по запуску (python-инструменты)

```bash
python -m venv .venv && .venv\Scripts\pip install -r tools/requirements.txt   # Windows
.venv\Scripts\python tools/eval_asr/eval_asr.py --help
```

Примеры использования инструментов — в `tools/TOOLS.md`.
