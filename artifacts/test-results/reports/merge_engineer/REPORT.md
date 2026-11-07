# merge-engineer: слияние ASR-сегментов с говорящими (baseline-0.1)

Дата: 2026-11-06
Окружение: Windows, `D:\Downloads_D\nuclear_hack`, Python 3.12.10 (`.venv\Scripts\python.exe`)
Вход (фиксирован, модели/пороги не менялись):
- ASR: `sherpa-onnx-small-zipformer-ru-2024-09-18` int8,
  `tools/eval_asr/hyp/small_zipformer_ru/timestamps.json` (+ `<id>.txt`);
- Диаризация: конфиг A (CAM++ + pyannote segmentation-3.0 + FastClustering thr 0.7),
  `reports/diar_engineer/hyp/A_campplus_thr0.7/<id>.rttm`;
- Манифест: `data/test/manifest.jsonl` (13 записей, `duration_sec`).
Выход: `output/baseline/<id>.json` — 13 файлов по OUTPUT CONTRACT из `rules.md`.

## 1. Алгоритм слияния

Для каждого ASR-сегмента `[s, e]` и каждого RTTM-сегмента `[rs, re]` спикера `sp`:

```
overlap(sp) = Σ_{r ∈ RTTM(sp)} max(0, min(e, re) − max(s, rs))
```

1. Спикер = `argmax_sp overlap(sp)`; при равенстве — лексикографически меньший `id`
   (детерминированность).
2. Если `overlap = 0` у всех спикеров (пересечения нет), берётся спикер ближайшего
   по времени RTTM-сегмента: `distance = max(0, max(rs − e, s − re))`, при равенстве —
   меньший `start`, затем меньший `id`. Это fallback, считается отдельно.
3. `start`/`end` — из ASR-таймингов (float, округление до 3 знаков). Сегменты
   сортируются по `start` (затем `end`, `speaker`).
4. `speakers` — уникальные id из RTTM, `label = null`. `confidence = 0.0` —
   **не измерено**: ASR-модель не отдаёт уверенность.
5. Нулевые по длине ASR-сегменты (`end == start`) расширяются на `0.01` с, чтобы
   выполнить контракт `start < end`. Причина: `words[].end` в `timestamps.json` —
   это время начала последнего токена, а не конца звучания (см. отчёт asr-engineer).

Инструмент: `tools/run_pipeline/merge.py` 0.1. Одна команда:

```powershell
.venv\Scripts\python.exe tools\run_pipeline\merge.py `
  --asr-hyp tools\eval_asr\hyp\small_zipformer_ru `
  --rttm reports\diar_engineer\hyp\A_campplus_thr0.7 `
  --manifest data\test\manifest.jsonl --out output\baseline --validate
```

Проверка только существующих файлов:

```powershell
.venv\Scripts\python.exe tools\run_pipeline\merge.py --out output\baseline --validate-only
```

## 2. Статистика

| id | спикеров (RTTM) | сегментов ASR | без пересечения (fallback) | малое пересечение (<0.1 с) | расширено (end==start) |
|---|---:|---:|---:|---:|---:|
| ruls_0001 | 1 | 8 | 0 | 0 | 0 |
| ruls_0002 | 1 | 5 | 0 | 0 | 0 |
| ruls_0003 | 1 | 10 | 0 | 1 | 1 |
| ruls_0004 | 1 | 5 | 0 | 0 | 0 |
| ruls_0005 | 1 | 1 | 0 | 0 | 0 |
| ruls_0006 | 1 | 6 | 0 | 1 | 1 |
| ruls_0007 | 1 | 2 | 0 | 0 | 0 |
| ruls_0008 | 1 | 8 | 0 | 0 | 0 |
| ruls_0009 | 1 | 5 | 0 | 0 | 0 |
| ruls_0010 | 1 | 6 | 0 | 0 | 0 |
| diar_0001 | 2 | 12 | 1 | 1 | 2 |
| diar_0002 | 3 | 8 | 1 | 0 | 1 |
| diar_0003 | 5 (ref 4) | 8 | 0 | 0 | 0 |
| **итого** | — | **84** | **2** | **3** | **5** |

- Всего сегментов: **84**; fallback без пересечения: **2** (2.4%);
  малое пересечение 0 < overlap < 0.1 с: **3** (3.6%).
- `diar_0003`: RTTM конфига A содержит 5 спикеров (`SPEAKER_1..4, SPEAKER_6`) при
  референсе 4 — это результат диаризации, слияние его не исправляет и не скрывает.
- Односпикеровые записи (10 шт.) получили ровно одного спикера — ложных спикеров
  на уровне слияния нет (они пришли бы только из RTTM).

## 3. Валидация OUTPUT CONTRACT

Встроенный режим `--validate` (после слияния) и `--validate-only` проверяет:
набор и типы top-level ключей, `language == "ru"`, `duration_sec > 0`, `speakers`
(уникальные строковые `id`, `label = null`), `segments` (ровно ключи
`speaker/start/end/text/confidence`, `speaker` из списка `speakers`, числовые
`start/end`, `start < end`, `start >= 0`, порядок по `start`, `text` — строка,
`confidence` — число или null), `meta` (ровно `asr_model/diar_model/version`, строки),
а также совпадение `audio_id` с именем файла.

Результат: **13/13 PASS**, код возврата 0.

Воспроизводимость: повторный прогон в отдельный каталог дал **0 отличий** по SHA256
для всех 13 файлов. Запуск без `--force` при существующих файлах — код возврата 2
(перезаписи нет).

## 4. Пример JSON (ruls_0007, сокращён текст)

```json
{
  "audio_id": "ruls_0007",
  "duration_sec": 19.39,
  "language": "ru",
  "speakers": [{"id": "SPEAKER_1", "label": null}],
  "segments": [
    {"speaker": "SPEAKER_1", "start": 0.0, "end": 8.08,
     "text": "но робкой горести моей ...", "confidence": 0.0},
    {"speaker": "SPEAKER_1", "start": 8.88, "end": 18.92,
     "text": "и все мне дико мрачно стало ...", "confidence": 0.0}
  ],
  "meta": {
    "asr_model": "sherpa-onnx-small-zipformer-ru-2024-09-18 int8",
    "diar_model": "campplus+pyannote-seg-3.0+FastClustering(thr=0.7)",
    "version": "baseline-0.1"
  }
}
```

## 5. Ограничения

1. **Только временное пересечение.** Спикер назначается по максимуму пересечения;
   перебивания не восстанавливаются, потому что в ASR-таймингах нет меток
   перекрытия, а RTTM-сегменты синтетического набора последовательные (в ref
   0 overlap-кадров, см. отчёт diar-engineer). Пересечения сегментов в выходе
   возможны только там, где они уже есть в исходных ASR-таймингах.
2. **ASR нарезался по 20 с** (ограничение модели). На границах кусков возможны
   потерянные/склеенные слова; слияние это не исправляет.
3. **`confidence` не измерена.** ASR-модель не отдаёт уверенность, поэтому во всех
   сегментах `0.0` — это заглушка, а не оценка качества.
4. **Нулевые сегменты.** 5 сегментов имели `end == start` из-за семантики
   `words[].end`; они расширены на 0.01 с. Это минимальная правка для контракта,
   а не уточнение таймингов.
5. **Качество спикеров наследуется от диаризации.** DER конфига A = 0.480 на 3
   многоговорящих записях; слияние не улучшает и не ухудшает разметку спикеров,
   только переносит её на ASR-сегменты.
6. **Мало данных.** 13 записей, из них 10 односпикеровых; 3 многоговорящие —
   синтетический TTS. Статистика слияния на реальных разговорах **не измерена**.
7. **`diar_0003`**: 5 гипотезных спикеров против 4 в референсе — известное поведение
   FastClustering (см. отчёт diar-engineer), не ошибка слияния.

## 6. Артефакты

- JSON: `output/baseline/<id>.json` (13 файлов).
- Скрипт: `tools/run_pipeline/merge.py` (версия 0.1).
- Реестр: `tools/TOOLS.md` (запись `run_pipeline (merge.py)`).
- Журнал: `logs/journal.md` (раздел merge-engineer + пометка оркестратора).
