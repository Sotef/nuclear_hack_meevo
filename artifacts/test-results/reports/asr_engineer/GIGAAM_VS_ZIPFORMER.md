# GigaAM v3 RNN-T vs small-zipformer-ru — сравнение ASR

Дата: 2026-11-07. Набор: `data/test` (13 wav, 310.641 с, manifest `data/test/manifest.jsonl`).
Инструмент WER: `tools/eval_asr/eval_asr.py` (одна и та же нормализация: lowercase, ё→е, пунктуация удалена).
Декодирование: greedy. Потоки: 1. Референсы и manifest не менялись.

## Лицензионный гейт

- Источник: `DaniilYemelyanenka/rashifruy`, commit `baa40b5e893b906762ae420a882ef50db031661d`, папка `src/mobile/android/app/src/main/assets/gigaam/`.
- Upstream: `salute-developers/GigaAM`, **MIT** (`LICENSE`, Copyright (c) 2024 GigaChat Team).
- Метаданные encoder ONNX: `comment=v3`, `version=3`, `model_type=EncDecRNNTBPEModel`, `is_giga_am=1`, `license=https://github.com/salute-developers/GigaAM/blob/main/LICENSE`.
- **Вердикт: v3 подтверждён, MIT. Гейт пройден.** (v2/NC не используется.)

## WER (micro / macro)

| модель | WER micro | WER macro | S | D | I | errors / ref |
|---|---:|---:|---:|---:|---:|---:|
| small-zipformer-ru (int8) | 0.146881 | 0.144337 | 49 | 14 | 10 | 73 / 497 |
| **GigaAM v3 RNN-T (int8 enc)** | **0.120724** | **0.101869** | 30 | 27 | 3 | 60 / 497 |

Гипотезы: `tools/eval_asr/hyp/small_zipformer_ru/` (существующие, сверены) и `tools/eval_asr/hyp/gigaam_v3/` (новые).
Отчёты: `reports/asr_engineer/wer_zipformer.{json,txt}`, `reports/asr_engineer/wer_gigaam_v3.{json,txt}`.

### 3 худших / 3 лучших файла (WER)

| модель | худшие | лучшие |
|---|---|---|
| zipformer | diar_0003 0.2857, ruls_0010 0.2222, ruls_0005 0.2045 | ruls_0006 0.0000, ruls_0004 0.0571, ruls_0007/0008 0.0714 |
| GigaAM v3 | diar_0001 0.3621, diar_0003 0.2381, diar_0002 0.1923 | ruls_0001 0.0000, ruls_0008 0.0000, ruls_0006 0.0294 |

GigaAM лучше на коротких `ruls_*` (аудиокниги), но хуже на длинных `diar_*`: больше удалений (D=27 vs 14), особенно diar_0001 (13 удалений). Это согласуется с ограничением «деградация на длинных чанках».

## Скорость и размер

| метрика | small-zipformer-ru | GigaAM v3 RNN-T |
|---|---:|---:|
| размер комплекта (int8) | ~25.7 MiB (весь каталог 28.64 MB) | **221.15 MiB** |
| RTF CPU (threads=1, 2 прохода) | 0.018417 | 0.088437 |
| RTF GPU | не измерен (sherpa pip CPU-only) | 0.081581 (encoder CUDA EP + decoder/joiner CPU) |
| пик RAM (CPU) | 338.8 MB | 473.23 MB |
| пик VRAM (GPU) | — | 3231 MB |
| чанкинг | 19 с | 60 с (родной лимит; модель ~200 с) |

- CPU: `D:\temp\opencode\bench_asr.py` (sherpa-onnx 2.0.1, `model_type=nemo`, `feature_dim=64` для GigaAM; 80 для zipformer), 2 прохода, среднее.
- GPU: sherpa-onnx pip-сборка без CUDA (`provider='cuda'` → fallback CPU). Использован `onnxruntime-gpu 1.31.1`: encoder на `CUDAExecutionProvider`, decoder/joiner на CPU, greedy RNN-T на Python. RTF GPU включает накладные расходы Python-цикла, поэтому ускорение мало (0.0884 → 0.0816).
- Кросс-проверка GPU-декода (torch mel-фичи): WER micro 0.100604 — близко к sherpa-прогону, подтверждает корректность декодера; официальный WER — sherpa (0.120724).

## Вердикт по гейту

Гейт: **WER ≤ 0.146881 И RTF ≤ 0.5**.

- WER: 0.120724 ≤ 0.146881 — **PASS** (относительное улучшение −17.8%).
- RTF: 0.088437 (CPU) и 0.081581 (GPU) ≤ 0.5 — **PASS**.
- **GigaAM v3 проходит гейт.**

## Рекомендация

- По качеству замена оправдана: WER −17.8% relative на одном наборе, лучше на коротких аудиокнижных фрагментах.
- Цена: комплект **221.15 MiB против ~25.7 MiB (×8.6)**, пик RAM +134 MB, VRAM ~3.2 GB (для mobile нерелевантно, для сервера ок).
- На длинных `diar_*` GigaAM даёт больше удалений (D=27 vs 14) — нужен контроль чанкинга ≤60 с и отдельный тест long-form.
- **Итог: не «слепая» замена.** Рекомендуется замена при приоритете качества и допустимом размере ~221 MiB; для size-constrained mobile — оставить small-zipformer (или GigaAM только с квантованием/урезанием). Перед заменой — A/B на целевом устройстве.

## Ограничения

- Набор — аудиокниги LibriVox (`ruls_*`) + 3 диаризационных файла; домен не равен реальным звонкам/шуму.
- Разная нарезка: zipformer 19 с, GigaAM 60 с — RTF и WER не полностью «при прочих равных».
- GPU-замер GigaAM — encoder CUDA + Python greedy; zipformer GPU не измерялся (нет GPU-пути в sherpa pip).
- WER macro чувствителен к 13 файлам; micro — основной.
