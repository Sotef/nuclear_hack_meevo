# merge-engineer: интеграция LLM-постпроцессора (baseline-0.2)

Дата: 2026-11-06
Окружение: Windows, `D:\Downloads_D\nuclear_hack`, Python 3.12.10 (`.venv\Scripts\python.exe`)
Вход (модели/пороги не менялись):
- ASR: `sherpa-onnx-small-zipformer-ru-2024-09-18` int8,
  `tools/eval_asr/hyp/small_zipformer_ru/timestamps.json` + `<id>.txt`;
- Диаризация: конфиг A (CAM++ + pyannote segmentation-3.0 + FastClustering thr 0.7),
  `reports/diar_engineer/hyp/A_campplus_thr0.7/<id>.rttm`;
- Манифест: `data/test/manifest.jsonl` (13 записей);
- LLM: `Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning` (int4-комплект —
  `models/llm/QVikhr-3-1.7B-Instruction-noreasoning-int4/`), промпт
  `prompts/postproc_v1.md` (`postproc_v1`).
Выход: `output/baseline_llm/<id>.json` — 13 файлов, `baseline-0.2`
(OUTPUT CONTRACT + additive-поля). `output/baseline/` (baseline-0.1) не изменялся.

## 1. LLM-прогон для 13 записей

Готовый прогон `reports/postproc_eval/llm_torch_correct13.jsonl` покрывает ровно
13 id (`diar_0001..3`, `ruls_0001..10`), его вход `text_asr` совпадает с
`tools/eval_asr/hyp/small_zipformer_ru/<id>.txt` для всех 13 записей. Это режим
`correct` (исправление ошибок ASR), но промпт `correct` **не просит пунктуацию**,
поэтому его вывод остаётся без знаков препинания.

Продукт-конфигурация по `POSTPROC_REPORT.md` — `punctuate` + консервативный
`correct`. Чтобы получить именно «пунктуация + исправления», поверх
`llm_torch_correct13.jsonl` выполнен один дополнительный прогон существующего
инструмента `run_llm` в режиме `punctuate`:

```powershell
.venv\Scripts\python.exe tools\run_llm\run_llm.py `
  --model C:\temp\opencode\qvikhr_src `
  --in reports\postproc_eval\llm_torch_correct13.jsonl `
  --out reports\postproc_eval\llm_torch_correct13_punct.jsonl `
  --mode punctuate --backend torch --device cuda `
  --metrics reports\postproc_eval\llm_torch_correct13_punct_metrics.json --force
```

- 13/13 элементов, GPU RTX 3080, torch 2.11.0+cu128, bf16, greedy;
  60.28 с всего, **4.637 с/элемент**.
- Итоговый LLM-текст = `punctuate(correct(raw))`. Модель для torch-прогона —
  HF-источник `C:\temp\opencode\qvikhr_src` (каталог `...-int4` — ONNX-only,
  torch его не загружает); в `meta.llm_model` записан продуктовый идентификатор
  int4-комплекта, как требует задание.
- `full`-режим **не использовался**: по `POSTPROC_REPORT.md` он ухудшает WER
  (+0.032193) и даёт 16 % false-alarm, поэтому не является дефолтом.

## 2. Алгоритм склейки `text_raw` → `text` и расчёт `edits`

LLM работает по записи целиком, а контракт требует текст по сегментам. Склейка
детерминированная, без модели и без случайности:

1. **Якорь.** Берутся тексты ASR-сегментов из `timestamps.json`; их слова
   конкатенируются в последовательность `anchor_words`, а `seg_word_counts[i]`
   хранит число слов сегмента `i`.
2. **`text_raw`.** Полный сырой текст `<id>.txt` разбивается на слова и
   выравнивается к `anchor_words` через `difflib.SequenceMatcher` по
   нормализованным словам (lowercase, `ё→е`, снятие пунктуации). Каждое слово
   `<id>.txt` получает сегмент своего выровненного якорного слова; вставки
   прикрепляются к ближайшему предыдущему якорю (в начале — к первому).
   `text_raw` сегмента = склейка назначенных слов. Проверено: конкатенация
   `text_raw` по сегментам **точно** восстанавливает `<id>.txt` для всех 13 записей.
3. **`text`.** Аналогично выравнивается текст LLM-вывода к тому же `anchor_words`;
   `text` сегмента = склейка назначенных LLM-слов (с пунктуацией/регистром).
   Проверено: конкатенация `text` по сегментам **точно** восстанавливает
   `llm_torch_correct13_punct.jsonl` для всех 13 записей.
4. **`edits`.** Для каждого сегмента — word-level diff `text_raw → text`
   (`difflib.SequenceMatcher` по точным словам, чтобы ловить и пунктуацию/регистр).
   Для каждой операции `replace/delete/insert` пишется `{from, to, reason}`:
   `from` — слова `text_raw`, `to` — слова `text` (пустая строка для delete/insert).
   `reason` берётся из поля `edits` LLM-вывода, если `from` совпадает; иначе
   `"punct"`. В режимах `correct`/`punctuate` структурированных `edits` нет,
   поэтому все 63 правки помечены `"punct"` (включая 9 словарных замен — это
   следствие fallback из задания, а не ошибка diff).
5. **`doubtful`.** `bool`: `true`, если в тексте сегмента встречается хотя бы один
   span из `doubtful` LLM-вывода. В `correct`/`punctuate` список пуст → все `false`.
6. **Спикеры/тайминги** — без изменений относительно baseline-0.1 (тот же
   `argmax` пересечения с RTTM, fallback по ближайшему сегменту, сортировка по
   `start`, `confidence = 0.0`).

## 3. Статистика интеграции

Всего: **84 сегмента**, **44 сегмента с edits** (52.4 %), **63 правки**,
**0 doubtful**, суммарная длина правок **940 символов**.

| id | сегментов | с edits | edits | doubtful | edit chars |
|---|---:|---:|---:|---:|---:|
| diar_0001 | 12 | 11 | 19 | 0 | 278 |
| diar_0002 | 8 | 6 | 8 | 0 | 135 |
| diar_0003 | 8 | 1 | 1 | 0 | 14 |
| ruls_0001 | 8 | 7 | 8 | 0 | 125 |
| ruls_0002 | 5 | 5 | 7 | 0 | 91 |
| ruls_0003 | 10 | 3 | 3 | 0 | 44 |
| ruls_0004 | 5 | 1 | 1 | 0 | 12 |
| ruls_0005 | 1 | 1 | 5 | 0 | 76 |
| ruls_0006 | 6 | 4 | 6 | 0 | 91 |
| ruls_0007 | 2 | 0 | 0 | 0 | 0 |
| ruls_0008 | 8 | 3 | 3 | 0 | 60 |
| ruls_0009 | 5 | 1 | 1 | 0 | 10 |
| ruls_0010 | 6 | 1 | 1 | 0 | 4 |
| **итого** | **84** | **44** | **63** | **0** | **940** |

Типы правок: `replace` — 62, `delete` — 1, `insert` — 0.
Из 63 правок: **54** меняют только пунктуацию/регистр (нормализованные `from`/`to`
совпадают), **9** меняют слова (коррекции ASR). Средняя длина правки —
**14.92 символа** (`from` 7.51 + `to` 7.41), в среднем **1.43 правки** на
сегмент с edits. 40 сегментов остались без правок.

`ruls_0007` — единственная запись без правок: `punctuate` вернул текст без
изменений (модель не расставила знаки), `correct` тоже ничего не изменил.

## 4. Валидация и детерминизм

| Проверка | Команда | Результат |
|---|---|---|
| v0.2 | `merge.py --out output\baseline_llm --validate-only` | **13/13 PASS**, exit 0 |
| v0.1 (регресс) | `merge.py --out output\baseline --validate-only` | **13/13 PASS**, exit 0 |
| v0.1 merge-регресс | merge без `--llm-*` в temp | **0 отличий SHA256** от `output/baseline` |
| Детерминизм v0.2 | повторный merge в `D:\temp\opencode\merge_llm_check` | **0 отличий SHA256** (13/13) |
| Защита от перезаписи | merge в `output\baseline_llm` без `--force` | exit **2**, файлы не изменены |

Валидатор v0.2 дополнительно проверяет: `segments[].text_raw` — строка,
`segments[].doubtful` — `bool`, `segments[].edits` — массив объектов ровно с
ключами `from/to/reason` (все строки), `meta.llm_model` и `meta.postproc_version`
— непустые строки. Для `baseline-0.1` схема остаётся строгой (ровно старые
ключи), поэтому регресс проходит.

## 5. Пример JSON (diar_0001, первый сегмент)

```json
{
  "audio_id": "diar_0001",
  "duration_sec": 46.132,
  "language": "ru",
  "speakers": [{"id": "SPEAKER_1", "label": null}],
  "segments": [
    {
      "speaker": "SPEAKER_1",
      "start": 0.0,
      "end": 2.8,
      "text": "Расскажи мне о сегодняшнем маршруте.",
      "confidence": 0.0,
      "text_raw": "расскажи мне о сегодняшнем маршруте",
      "doubtful": false,
      "edits": [
        {"from": "расскажи", "to": "Расскажи", "reason": "punct"},
        {"from": "маршруте", "to": "маршруте.", "reason": "punct"}
      ]
    }
  ],
  "meta": {
    "asr_model": "sherpa-onnx-small-zipformer-ru-2024-09-18 int8",
    "diar_model": "campplus+pyannote-seg-3.0+FastClustering(thr=0.7)",
    "version": "baseline-0.2",
    "llm_model": "Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning int4",
    "postproc_version": "postproc_v1 (punctuate+correct)"
  }
}
```

## 6. Ограничения

1. **`full`-режим не интегрирован дефолтом.** По `POSTPROC_REPORT.md` он даёт
   WER +0.032193, false-alarm 16 % и галлюцинации; для продукта выбраны
   `punctuate` + консервативный `correct`. `full` можно подключить тем же
   `--llm-hyp`, но это экспериментальный путь.
2. **Качество LLM-правок не переоценивается.** `punctuate`: punct F1 0.757,
   boundary F1 0.849; `correct`: WER 0.146881 → 0.138833 (delta −0.008048),
   improved 1/13, worsened 1/13. На 13 записях это статистически грубо.
   `ruls_0007` вообще не получил пунктуации.
3. **`reason` = `"punct"` для всех правок.** Режимы `correct`/`punctuate` не
   отдают структурированные `edits`, поэтому 9 словарных замен тоже помечены
   `"punct"` (fallback из задания). Для семантически точных reason нужен
   `full`-режим или отдельный структурированный вывод корректора.
4. **Границы сегментов.** LLM-текст распределяется по ASR-сегментам
   выравниванием слов; граница предложения может попасть на границу сегмента
   (например, точка приклеивается к последнему слову сегмента). Это не меняет
   тайминги и спикеров.
5. **`text_raw` — из `<id>.txt`, а не из `timestamps.json`.** Полный текст
   `<id>.txt` (склейка чанков ASR) на 7/13 записей немного отличается от
   конкатенации сегментных текстов `timestamps.json` (границы чанков/слов).
   Выравнивание к сегментам сохраняет `<id>.txt` точно, но отдельные слова
   могут попасть в соседний сегмент на стыке.
6. **`doubtful` не проверен на реальных флагах.** В `correct`/`punctuate` LLM
   не помечает сомнительные места, поэтому все 84 сегмента — `false`. Логика
   маппинга span→сегмент реализована, но на непустых флагах не обкатана.
7. **Модель/железо.** torch-прогон сделан на HF-источнике (int4-каталог —
   ONNX-only), GPU RTX 3080 bf16. Это не замер Android; мобильные оценки —
   в `POSTPROC_REPORT.md`.
8. **Мало данных.** 13 записей, 10 односпикеровых; 3 многоговорящие — TTS.
   Статистика правок на реальных разговорах не измерена.

## 7. Артефакты

- JSON: `output/baseline_llm/<id>.json` (13 файлов, `baseline-0.2`).
- LLM-вывод: `reports/postproc_eval/llm_torch_correct13_punct.jsonl`
  (+ `_metrics.json`); исходный `llm_torch_correct13.jsonl` не изменялся.
- Инструмент: `tools/run_pipeline/merge.py` 1.1.0
  (`--llm-hyp/--llm-model/--postproc-version`, валидация v0.1/v0.2).
- Реестр: `tools/TOOLS.md` (запись `run_pipeline (merge.py)`, версия 1.1.0).
- Журнал: `logs/journal.md` (раздел merge-engineer, baseline-0.2).
