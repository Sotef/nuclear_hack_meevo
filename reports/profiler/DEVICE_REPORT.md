# DEVICE_REPORT — замеры на Samsung SM-N9750 (Android 12, arm64-v8a)

Дата: 2026-11-07. Агент: profiler (с полномочиями optimizer). Устройство: SM-N9750, serial RF8N10DPSDL, Android 12, 12 CPU.
Все замеры — на устройстве через `adb shell`, wall-time = `date +%s%N` вокруг команды (обёртка `bench.sh`, ns→ms через awk, т.к. mksh 32-bit). RAM = `VmHWM` из `/proc/<pid>/status` (поллинг 50 мс во время прогона).

## 1. Статус рантаймов

| Стадия | Статус | Как запущено |
|---|---|---|
| ASR (sherpa-onnx small-zipformer-ru int8) | **РАБОТАЕТ** | `sherpa-onnx-offline` (termux-static v2.0.1) + `libhashshim.so` (LD_PRELOAD) |
| Диаризация (pyannote seg + 3dspeaker CAM++) | **РАБОТАЕТ** | `sherpa-onnx-offline-speaker-diarization` + `model.fixed.onnx` (топосорт+инициализаторы), threshold 0.7 |
| LLM (MiniCPM5-2B+LoRA int4) | **НЕ ИЗМЕРЕНО** | см. §6 |

### Что починено (блокеры прошлого раунда)
1. **CLI sherpa-onnx.** Скачан `sherpa-onnx-v2.0.1-android-aarch64-termux-static.tar.bz2` (158 МБ, Apache-2.0). Бинари статические по sherpa/ORT, но требуют `libc++_shared.so`. NDK r27c libc++ не экспортирует `std::__ndk1::__hash_memory` (удалён в новом libc++). Собран шим `device_bundle/src/hashshim.cpp` → `libhashshim.so` (NDK r27c, aarch64-linux-android24), подключается через `LD_PRELOAD`. Проверено: `--help` и реальные прогоны.
2. **/sdcard noexec.** Бинари нельзя запускать с `/sdcard` (Permission denied). Бинари+шим лежат в `/data/local/tmp/pipeline/bin/sherpa/`, модели/аудио — на `/sdcard/pipeline/`.
3. **pyannote segmentation отсутствовал в bundle.** Скачан `sherpa-onnx-pyannote-segmentation-3-0.tar.bz2` (MIT). Модели `model.onnx`/`model.int8.onnx` падают в ORT 1.22 с «Nodes in a graph must be topologically sorted» (If-субграф захватывает внешний initializer). Исправлено скриптом: топосорт графа с учётом implicit-capture + копирование захваченных initializer'ов внутрь субграфов → `model.fixed.onnx` (6.15 МБ) и `model.int8.fixed.onnx`. Работает.

## 2. Таблица замеров (1 холодный + 2 тёплых)

Длительности: ruls_0001 = 19.880 с, diar_0002 = 33.507 с, diar_0003 = 36.441 с.
`wall RTF` = wall_ms / длительность; `decode RTF` = внутренний таймер sherpa (без загрузки модели).

| Файл | Стадия | Прогон | wall, с | wall RTF | decode, с | decode RTF | VmHWM, МБ |
|---|---|---|---:|---:|---:|---:|---:|
| ruls_0001 | ASR | cold r1 | 8.710 | 0.438 | 0.764 | 0.038 | 306.1 |
| ruls_0001 | ASR | warm r2 | 8.331 | 0.419 | 0.799 | 0.040 | 306.2 |
| ruls_0001 | ASR | warm r3 | 8.826 | 0.444 | 0.768 | 0.039 | 306.3 |
| ruls_0001 | diar | cold r1 | 7.592 | 0.382 | 6.147 | 0.309 | 163.0 |
| ruls_0001 | diar | warm r2 | 7.492 | 0.377 | 6.200 | 0.312 | 162.7 |
| ruls_0001 | diar | warm r3 | 7.594 | 0.382 | 6.191 | 0.311 | 162.4 |
| diar_0002 | ASR | r1–r3 | — | — | — | — | — |
| diar_0002 | diar | cold r1 | 14.744 | 0.440 | 13.418 | 0.400 | 158.9 |
| diar_0002 | diar | warm r2 | 14.894 | 0.445 | 13.527 | 0.404 | 158.5 |
| diar_0002 | diar | warm r3 | 14.802 | 0.442 | 13.483 | 0.402 | 158.9 |
| diar_0003 | ASR | r1–r3 | — | — | — | — | — |
| diar_0003 | diar | cold r1 | 13.661 | 0.375 | 12.291 | 0.337 | 149.4 |
| diar_0003 | diar | warm r2 | 13.726 | 0.377 | 12.336 | 0.339 | 149.5 |
| diar_0003 | diar | warm r3 | 13.754 | 0.377 | 12.420 | 0.341 | 149.4 |

**ASR на diar_0002/0003 — не измерено:** модель small-zipformer-ru падает (`Ort::Exception`, Reshape `{1,672,16}` → `{-1,3343,4,4}`) на аудио >~20 с. Это известное ограничение модели (нужна нарезка на чанки ≤20 с; в CLI её нет). ruls_0001 (19.88 с) — на границе, проходит.

**Холодный vs тёплый:** разница в пределах шума (<6%). Загрузка моделей с `/sdcard` (FUSE) стабильно медленная: ASR ~7.5 с (`recognizer created in 7.539 s`), diar ~1.3–1.4 с. Поэтому wall RTF для ASR в основном = загрузка модели; decode RTF ~0.04.

## 3. Системные метрики

- **Пиковая RAM (VmHWM):** ASR ~306 МБ; diar 149–163 МБ (растёт с длительностью/числом спикеров).
- **Температура батареи:** после серии `dumpsys battery` → `temperature: 211` = **21.1 °C** (level 100%). До серии не снято (серия уже шла) — «до: не измерено».
- **Свободное место /sdcard:** 229 ГБ всего, 150 ГБ занято, **79 ГБ свободно** (66%). До/после серии не менялось (модели уже лежали).
- **Размер bundle на устройстве:** `/sdcard/pipeline` = **103 МБ** (ASR 25 МБ, diar 42 МБ, LLM 3.5 КБ — LLM не пушился). Полный bundle на хосте 1.62 ГБ (LLM 1.64 ГБ не пушился).

## 4. Оценка vs замер

| Стадия | Desktop RTF | Оценка mobile (POSTPROC_REPORT) | Замер (decode) | Разница vs desktop | Комментарий |
|---|---:|---:|---:|---:|---|
| ASR | 0.018949 | 0.095–0.19 (×5–10) | **0.038–0.040** | **×2.0–2.1** | лучше оценки (оценка была пессимистична) |
| Диаризация | 0.0775 | 0.388–0.775 (×5–10) | **0.309–0.404** | **×4.0–5.2** | на нижней границе оценки |

Wall RTF диаризации 0.375–0.445 — внутри/ниже оценки 0.388–0.775. Wall RTF ASR 0.42–0.44 включает разовую загрузку модели; в пайплайне recognizer создаётся один раз, поэтому релевантен decode RTF 0.04.

## 5. Вердикт по гейту RTF ≤ 1.0 (rules.md)

- ASR (ruls_0001): decode RTF 0.038–0.040, wall RTF 0.42–0.44 → **PASS**.
- Диаризация (все 3 файла): decode RTF 0.309–0.404, wall RTF 0.375–0.445 → **PASS**.
- LLM: **не измерено** (нет вердикта).

## 6. LLM — почему не измерено

- **Уровень (а) ORT GenAI:** установленная/последняя на PyPI версия `onnxruntime-genai` = **0.19.0** (новее нет). Она отвергает `tokenizer.json` MiniCPM5 (regex `\p{L}`/`\p{N}`); после ASCII-замены — `invalid string_view position`. Android-бинарь `llm_cli` не собран (нужен AAR `onnxruntime-genai-android` + подтверждённый фикс токенизатора).
- **Уровень (б) LiteRT:** официальный `litert-community/MiniCPM5-2B` int4 (1.55 ГБ, `.litertlm`) доступен, но релизы `google-ai-edge/LiteRT-LM` v0.19.1 содержат только macOS/iOS бинари (`litert_lm_main.macos_arm64`, `CLiteRTLM.xcframework`); **Android-prebuilt нет**, `bazel` в окружении отсутствует → собрать `litert_lm_main` под arm64 Android нечем. Модель не скачивалась (нет рантайма).
- Итог: **«LLM на устройстве: не измерено»** — оба уровня лестницы заблокированы инфраструктурно.

## 7. Артефакты

- `reports/profiler/device_runs/out_*.txt` — stdout всех 18 прогонов (ASR JSON, diar RTTM-подобный вывод, ошибки).
- `device_bundle/bin/sherpa/` — `sherpa-onnx-offline`, `sherpa-onnx-offline-speaker-diarization`, `libhashshim.so`, `.so`.
- `device_bundle/src/hashshim.cpp` — исходник шима.
- `device_bundle/models/diar/pyannote-segmentation-3.0/model.fixed.onnx`, `model.int8.fixed.onnx`.
- `device_bundle/scripts/bench.sh`, `measure_all.sh` — device-side замер/матрица.
- `device_bundle/scripts/push_to_device.ps1`, `run_stage.ps1` — обновлены под рабочие пути.

## 8. Лицензии скачанного

- sherpa-onnx v2.0.1 (termux-static) — Apache-2.0.
- pyannote segmentation-3.0 — MIT (LICENSE в `models/diar/pyannote-segmentation-3.0/`).
- 3dspeaker CAM++ — Apache-2.0 (уже в bundle).
- NDK r27c libc++_shared.so — Apache-2.0 with LLVM exceptions.

## 9. ASR chunked (diar_0002 / diar_0003) — второй раунд

Причина: small-zipformer-ru падает (Reshape) на аудио >~20 с. Добавлен режим `-Chunked` в `device_bundle/scripts/run_stage.ps1`: PC-сторона режет WAV на куски ≤19 с (`.venv` + soundfile), пушит на `/sdcard/pipeline/audio/chunks/`, гоняет ASR по каждому куску через device-side `bench.sh` (wall + VmHWM), склеивает текст и суммирует время. diar_0002 = 2 куска (19.0 + 14.5 с), diar_0003 = 2 куска (19.0 + 17.4 с).

| Файл | Прогон | wall, с | wall RTF | decode, с | decode RTF | VmHWM, МБ |
|---|---|---:|---:|---:|---:|---:|
| diar_0002 (33.507 с) | cold r1 | 17.392 | 0.5191 | 1.273 | 0.0380 | 298.4 |
| diar_0002 | warm r2 | 17.353 | 0.5179 | 1.282 | 0.0383 | 298.9 |
| diar_0002 | warm r3 | 16.920 | 0.5050 | 1.274 | 0.0380 | 298.1 |
| diar_0003 (36.441 с) | cold r1 | 17.196 | 0.4719 | 1.391 | 0.0382 | 299.0 |
| diar_0003 | warm r2 | 17.318 | 0.4752 | 1.387 | 0.0381 | 298.8 |
| diar_0003 | warm r3 | 17.275 | 0.4741 | 1.421 | 0.0390 | 298.3 |

Wall RTF 0.47–0.52 включает повторную загрузку модели на каждый чанк (~7.5 с × 2). Decode RTF 0.038–0.039 — чистое ASR-время, совпадает с ruls_0001 (0.038–0.040). В одно-процессном пайплайне recognizer создаётся один раз, поэтому релевантен decode RTF.

## 10. LLM QVikhr-3-1.7B-Instruction-noreasoning int4 — замер на устройстве

Собран `llm_cli` (NDK r27c, aarch64-linux-android24, ORT GenAI 0.19.0 Android AAR + ONNX Runtime 1.23.0 Android AAR). Модель `models/llm/QVikhr-...-int4` (1.10 ГБ) запушена на `/sdcard/pipeline/models/llm/`. Промпт — `postproc_v1` режим `punctuate` (ChatML QVikhr, `/no_think`), greedy (`do_sample=false`), `max_new_tokens=256`. 1 холодный + 2 тёплых прогона на файл; wall = `bench.sh`, RAM = VmHWM.

| Файл | Прогон | wall, с | wall RTF | gen, с | decode RTF | ток/с | gen токенов | VmHWM, МБ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ruls_0001 (19.880 с) | cold r1 | 12.770 | 0.6424 | 4.502 | 0.2265 | 15.99 | 72 | 1416.7 |
| ruls_0001 | warm r2 | 12.790 | 0.6434 | 4.538 | 0.2283 | 15.87 | 72 | 1416.8 |
| ruls_0001 | warm r3 | 12.833 | 0.6455 | 4.528 | 0.2278 | 15.90 | 72 | 1416.7 |
| diar_0002 (33.507 с) | cold r1 | 20.344 | 0.6072 | 11.103 | 0.3314 | 15.13 | 168 | 1479.5 |
| diar_0002 | warm r2 | 20.276 | 0.6051 | 11.129 | 0.3321 | 15.10 | 168 | 1480.4 |
| diar_0002 | warm r3 | 20.383 | 0.6083 | 11.169 | 0.3333 | 15.04 | 168 | 1479.5 |
| diar_0003 (36.441 с) | cold r1 | 12.312 | 0.3379 | 4.205 | 0.1154 | 15.93 | 67 | 1409.7 |
| diar_0003 | warm r2 | 12.384 | 0.3398 | 4.185 | 0.1148 | 16.01 | 67 | 1409.7 |
| diar_0003 | warm r3 | 12.465 | 0.3421 | 4.195 | 0.1151 | 15.97 | 67 | 1410.1 |

`decode RTF` = gen_ms / длительность аудио. `wall` включает загрузку модели ~6.0 с (load_ms 5931–6093). Генерация корректная (пример ruls_0001: «Пред нею все побежала. Все вокруг вдруг опустела...»); модель всё же печатает пустой ` thinking...</think>` перед ответом (учтён в gen токенах). RAM 1.41–1.48 ГБ (модель 1.10 ГБ + ORT + KV-cache).

## 11. Сводная таблица полного пайплайна per file

Средние по 3 прогонам. `wall` — сумма измеренных wall-времён стадий (каждая стадия — отдельный процесс, поэтому включает загрузку моделей). `decode` — сумма внутренних таймеров (ASR decode + diar decode + LLM gen), т.е. чистое вычисление.

| Файл | ASR wall | diar wall | LLM wall | Σ wall | wall RTF | ASR dec | diar dec | LLM gen | Σ dec | dec RTF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ruls_0001 (19.880 с) | 8.622 | 7.559 | 12.798 | 28.979 | 1.458 | 0.777 | 6.179 | 4.523 | 11.479 | 0.577 |
| diar_0002 (33.507 с) | 17.222 | 14.813 | 20.334 | 52.369 | 1.563 | 1.276 | 13.476 | 11.134 | 25.886 | 0.773 |
| diar_0003 (36.441 с) | 17.263 | 13.714 | 12.387 | 43.364 | 1.190 | 1.400 | 12.349 | 4.195 | 17.944 | 0.492 |

## 12. Вердикт гейта RTF ≤ 1.0

- ASR: decode RTF 0.038–0.039, wall RTF 0.47–0.52 → **PASS**.
- Диаризация: decode RTF 0.31–0.40, wall RTF 0.375–0.445 → **PASS**.
- LLM: decode RTF 0.115–0.333, wall RTF 0.338–0.646 → **PASS**.
- Пайплайн по decode-сумме: 0.49–0.77 → **PASS**.
- Пайплайн по wall-сумме: 1.19–1.56 → **FAIL** из-за загрузки моделей в каждом процессе (ASR ~7.5 с/чанк, LLM ~6.0 с, diar ~1.3 с). В одно-процессном пайплайне с однократной загрузкой моделей per-file RTF = decode-сумма (0.49–0.77) → PASS.

## 13. Оценка vs замер (LLM) и пометка про фолбэк

| Метрика | Оценка (POSTPROC_REPORT) | Замер (QVikhr int4) |
|---|---:|---:|
| сек / мин аудио | ~38 | **6.9–19.9** (ruls_0001 13.6, diar_0002 19.9, diar_0003 6.9) |
| ток/с (мобиль) | 10–25 | **15.0–16.0** |

Оценка была консервативной: замер в 2–5× быстрее по сек/мин; ток/с — в середине диапазона.

**Фолбэк QVikhr:** QVikhr-3-1.7B-Instruction-noreasoning int4 использован ТОЛЬКО для device-замеров вместо MiniCPM5-smoke (у которого на устройстве не собирался рантайм). Качество: punct F1 QVikhr **0.757** vs MiniCPM5-smoke **0.782** (источник: `reports/postproc_eval/COMPARE.md`). Продуктовый выбор LLM остаётся открытым.

**Температура батареи:** после серии `dumpsys battery` → `temperature: 222` = **22.2 °C** (level 100%). До серии — «не измерено» (серия уже шла). Свободно на /sdcard: 78 ГБ.

**Лицензии:** ORT GenAI 0.19.0 Android AAR — MIT; ONNX Runtime 1.23.0 Android AAR — MIT; NDK r27c libc++_shared — Apache-2.0 with LLVM exceptions.
