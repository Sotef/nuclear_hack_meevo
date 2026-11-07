# diar-engineer: baseline диаризации (эмбеддинги + кластеризация)

Дата: 2026-11-06
Окружение: Windows, `D:\Downloads_D\nuclear_hack`, Python 3.12.10 (`.venv\Scripts\python.exe`)
Рантайм: `sherpa-onnx` 2.0.1 (`sherpa-onnx-core` 2.0.1), CPU, 1 поток
Оценка: `tools/eval_diar` 1.0.1, collar 0.25 с, шаг кадра 10 мс

## 1. Итог

| Конфигурация | Изменяемая переменная | DER (3 многоговорящих) | Overlap-DER | False alarms (10 односпикеровых) | RTF (wall/audio) |
|---|---|---:|---:|---:|---:|
| **A: CAM++ + threshold 0.7** | — (основная) | **0.480088** | не измерено (0 overlap-кадров в ref) | **0/10** | **0.0775** |
| **B: CAM++ + oracle num_speakers** | способ задания числа спикеров (threshold → oracle из manifest) | 0.551991 | не измерено (0 overlap-кадров в ref) | 0/10 | 0.0770 |
| C (диагностика, вне A/B): TitaNet-small + threshold 0.7 | модель эмбеддингов | 0.503429 | не измерено | не запускалось на 10 | 0.0739 (только 3 diar) |

Вывод: **A лучше B** на одном наборе и одном eval-инструменте (DER 0.480 vs 0.552).
Oracle-число спикеров не помогло: на `diar_0003` кластеризация с `num_clusters=4`
всё равно вернула 3 спикера и дала DER 0.812 (против 0.573 у A с 5 кластерами).
CAM++ лучше TitaNet-small на этом наборе (C: 0.503).

## 2. DER по записям (collar 0.25)

| id | Спикеров ref | A: hyp | A: DER | A: miss | A: FA | A: spk_err | B: hyp | B: DER | B: miss | B: FA | B: spk_err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| diar_0001 | 2 | 2 | 0.500528 | 1178 | 1 | 717 | 2 | 0.500528 | 1178 | 1 | 717 |
| diar_0002 | 3 | 3 | 0.351160 | 748 | 44 | 101 | 3 | 0.351160 | 748 | 44 | 101 |
| diar_0003 | 4 | 5 | 0.572536 | 1126 | 50 | 375 | 3 | 0.812477 | 1126 | 50 | 1025 |
| **micro** | — | — | **0.480088** | 3052 | 95 | 1193 | — | **0.551991** | 3052 | 95 | 1843 |

Единицы: speaker-time кадры по 10 мс; всего 9040 ref-кадров (90.4 с речи).
`miss` одинаков у A и B (3052) — ошибки пропуска определяются сегментацией, не кластеризацией.

### Overlap-DER

В референсных RTTM **нет кадров с перекрытием** (`overlap_total_ref_frames = 0`),
поэтому overlap-DER на этом наборе **не измерен** (инструмент возвращает 0.0 при
нулевом знаменателе). Это ограничение синтетического набора: сегменты спикеров
последовательные, без реальных пересечений. Отдельный handcrafted overlap-тест
`eval_diar` был проверен data-eval ранее (см. `reports/data_eval/REPORT.md`).

## 3. Ложные спикеры на односпикеровых записях

| Конфигурация | Записей с ref=1 | Получили >1 спикера | Доля |
|---|---:|---:|---:|
| A (threshold 0.7) | 10 | 0 | 0/10 |
| B (oracle) | 10 | 0 | 0/10 |

На всех 10 записях RuLS обе конфигурации вернули ровно 1 спикера. Ложных
срабатываний кластеризации на односпикеровом материале нет.

## 4. Параметры

| Параметр | A | B |
|---|---|---|
| Сегментация | `sherpa-onnx-pyannote-segmentation-3-0/model.onnx` (fp32, MIT) | то же |
| Эмбеддинги | `3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx` (Apache-2.0) | то же |
| Кластеризация | `FastClustering`, `threshold=0.7`, `num_clusters=-1` | `FastClustering`, `num_clusters=speakers_count` из manifest (oracle) |
| `min_duration_on` | 0.3 с | 0.3 с |
| `min_duration_off` | 0.5 с | 0.5 с |
| Потоки / провайдер | 1 / cpu | 1 / cpu |
| Окно эмбеддинга | внутреннее окно sherpa-onnx (не настраивалось) | то же |

Порог 0.7 выбран по короткому sweep на 3 diar-записях (только для выбора A,
не является отдельной конфигурацией):

| threshold | DER | hyp speakers (diar_0001/0002/0003) |
|---:|---:|---|
| 0.3 | 0.580310 | 5 / 5 / 9 |
| 0.5 | 0.540929 | 3 / 5 / 6 |
| **0.7** | **0.480088** | **2 / 3 / 5** |
| 0.9 | 0.651770 | 1 / 2 / 3 |

## 5. RTF инференса

Замер: `time.perf_counter()` вокруг `OfflineSpeakerDiarization.process()` (без
загрузки модели и чтения wav), CPU, 1 поток.

| Конфигурация | Суммарное аудио | Суммарный wall | RTF |
|---|---:|---:|---:|
| A (13 записей) | 310.64 с | 24.08 с | 0.0775 |
| B (13 записей) | 310.64 с | 23.92 с | 0.0770 |
| C (3 diar) | 116.08 с | 8.58 с | 0.0739 |

Это Python/CPU-замер на десктопе, **не** замер на Android. Для решения о
производительности на телефоне нужен profiler.

## 6. Версии

- Python 3.12.10
- sherpa-onnx 2.0.1, sherpa-onnx-core 2.0.1
- numpy 2.5.4, soundfile 0.14.0
- pyannote segmentation-3.0 (ONNX, fp32), CAM++ (ONNX), TitaNet-small (ONNX)
- eval_diar 1.0.1, run_diar 1.0.0

## 7. Команды запуска

```powershell
# A: CAM++ + threshold 0.7 (все 13 записей)
.venv\Scripts\python.exe tools\run_diar\run_diar.py --manifest data\test\manifest.jsonl `
  --out reports\diar_engineer\hyp\A_campplus_thr0.7 --threshold 0.7 `
  --report reports\diar_engineer\run_A.json --force

# B: CAM++ + oracle num_speakers из manifest
.venv\Scripts\python.exe tools\run_diar\run_diar.py --manifest data\test\manifest.jsonl `
  --out reports\diar_engineer\hyp\B_campplus_oracle --oracle-num-speakers `
  --report reports\diar_engineer\run_B.json --force

# DER (только 3 записи с ref RTTM)
.venv\Scripts\python.exe tools\eval_diar\eval_diar.py --ref data\test `
  --hyp reports\diar_engineer\hyp\A_campplus_thr0.7 --collar 0.25 `
  --report reports\diar_engineer\der_A
.venv\Scripts\python.exe tools\eval_diar\eval_diar.py --ref data\test `
  --hyp reports\diar_engineer\hyp\B_campplus_oracle --collar 0.25 `
  --report reports\diar_engineer\der_B_campplus_oracle
```

## 8. Ограничения

- **Синтетика.** 3 многоговорящие записи — TTS (`ivkond/synthetic-speech-diarization-ru`, MIT),
  а не живые люди; нет шума, дальнего микрофона, телефонного канала.
- **Нет реальных перекрытий.** RTTM-сегменты последовательные; overlap-DER не измерен.
- **Reference включает паузы.** RTTM размечает целые реплики, включая внутренние паузы TTS;
  сегментатор такие паузы не считает речью → завышенный `miss` (3052/9040 ≈ 34%).
  Это главный вклад в DER, а не ошибка кластеризации.
- **Мало данных.** 3 записи (2/3/4 спикера), 116 с; выводы о качестве статистически слабые.
- **Язык эмбеддингов.** CAM++ обучен на китайском, TitaNet — на английском; на русском
  проверено только этим замером.
- **Число спикеров.** В A оценивается порогом (без знания числа); в B используется oracle
  из manifest — это честно помечено и не является «слепой» конфигурацией.
- **RTF** — десктопный Python/CPU, не Android; для целевого устройства не измерялся.
- `diar_0003` в B: `num_clusters=4`, но sherpa вернул 3 спикера — поведение FastClustering
  при малом числе сегментов; требует отдельного разбора, если oracle-режим понадобится.

## 9. Рекомендация baseline

Брать **конфигурацию A**: CAM++ + pyannote segmentation-3.0 + FastClustering threshold 0.7.
Она даёт лучший DER (0.480), 0/10 ложных спикеров и RTF 0.078 на десктопе. Порог 0.7
следует перепроверить на реальных русских многоговорящих записях с перекрытиями;
до этого считать результат baseline, а не «достаточно хорошо» (гейт Т2 не задан).
