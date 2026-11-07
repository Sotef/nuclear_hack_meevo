# Reviewer: независимая проверка цикла LLM-постпроцессора

Дата: 2026-11-06. Автор: reviewer.
Окружение: Windows, `D:\Downloads_D\nuclear_hack`, Python 3.12.10
(`.venv\Scripts\python.exe`). LLM-генерация не запускалась — проверялись
готовые артефакты и eval-инструменты. Артефакты других агентов не изменялись;
все повторные прогоны писались в `D:\temp\opencode\review_llm*`.

Проверяемые отчёты: `reports/asr_engineer/POSTPROC_REPORT.md`,
`reports/merge_engineer/REPORT_LLM.md`, `models/llm/MANIFEST.md`,
`docs/ml_api_contract.md`, `data/postproc/build_stats.json`.

## 1. Перепроверка метрик `tools/eval_postproc` 1.0.0

Команды (повторяют заявленные прогоны, отчёты — во временный каталог):

```powershell
.venv\Scripts\python.exe tools\eval_postproc\eval_postproc.py --hyp reports\postproc_eval\llm_torch_punct300.jsonl --ref-punct data\postproc\punct_eval.jsonl --ref-corr data\postproc\correction_eval.jsonl --report D:\temp\opencode\review_llm\punct300.json --force
.venv\Scripts\python.exe tools\eval_postproc\eval_postproc.py --hyp reports\postproc_eval\llm_torch_correct13.jsonl --ref-punct data\postproc\punct_eval.jsonl --ref-corr data\postproc\correction_eval.jsonl --report D:\temp\opencode\review_llm\correct13.json --force
.venv\Scripts\python.exe tools\eval_postproc\eval_postproc.py --hyp reports\postproc_eval\llm_torch_full63.jsonl --ref-punct data\postproc\punct_eval.jsonl --ref-corr data\postproc\correction_eval.jsonl --report D:\temp\opencode\review_llm\full63.json --force
```

Все три прогона — exit 0. Дополнительно сверены сохранённые артефакты
ONNX/torch на 10 примерах (без запуска LLM).

| Метрика | Заявлено | Перепроверено | Совпадение |
|---|---:|---:|:---:|
| punct300 micro P | 0.8179 | 0.817875 | да |
| punct300 micro R | 0.7039 | 0.703919 | да |
| punct300 micro F1 | 0.7566 | 0.756630 | да |
| punct300 boundary F1 | 0.8486 | 0.848580 | да |
| punct300 TP/FP/FN | 485/108/204 | 485/108/204 | да |
| correct13 WER before | 0.146881 | 0.146881 | да |
| correct13 WER after | 0.138833 | 0.138833 | да |
| correct13 delta WER | −0.008048 | −0.008048 | да |
| correct13 improved/worsened/unchanged | 1/1/11 | 1/1/11 | да |
| full63 WER before | 0.146881 | 0.146881 | да |
| full63 WER after | 0.179074 | 0.179074 | да |
| full63 delta WER | +0.032193 | +0.032193 | да |
| full63 false-alarm (clean-50) | 0.16 | 0.160000 (8/50) | да |
| full63 punct F1 (clean-50) | 0.443038 | 0.443038 | да |
| full63 boundary F1 (clean-50) | 0.202899 | 0.202899 | да |
| full63 error coverage | 0.025210 | 0.025210 (3/119) | да |
| full63 pair coverage | 0.166667 | 0.166667 (2/12) | да |
| full63 flag density | 1.7316 / 100 слов | 1.732 / 100 слов (8/462) | да |
| ONNX int4 exact vs torch (10) | 3/10 | 3/10 | да |
| ONNX int4 mean char-similarity | 0.9856 | 0.9856 | да |
| ONNX int4 punct F1 (10) | 0.681818 | 0.681818 | да |
| ONNX int4 boundary F1 (10) | 0.720000 | 0.720000 | да |
| torch punct10 F1 / boundary F1 | 0.761905 / 0.833333 | 0.761905 / 0.833333 | да |

**Вывод:** все заявленные метрики воспроизводятся на готовых гипотезах.
Расхождений нет.

## 2. Контракт v0.2 и регресс v0.1

```powershell
.venv\Scripts\python.exe tools\run_pipeline\merge.py --out output\baseline_llm --validate-only
.venv\Scripts\python.exe tools\run_pipeline\merge.py --out output\baseline --validate-only
```

| Проверка | Результат |
|---|---|
| `output/baseline_llm` (v0.2) | **13/13 PASS**, exit 0 |
| `output/baseline` (v0.1) | **13/13 PASS**, exit 0 |

Ручная сверка `output/baseline_llm/diar_0001.json` (многоговорящий, 2 спикера,
12 сегментов) с `OUTPUT CONTRACT` и `docs/ml_api_contract.md`:

- базовые поля на месте: `audio_id`, `duration_sec`, `language`, `speakers[]`
  (`id`/`label`), `segments[]` (`speaker`/`start`/`end`/`text`/`confidence`),
  `meta` (`asr_model`/`diar_model`/`version`);
- additive-поля на месте: `segments[].text_raw`, `segments[].doubtful`,
  `segments[].edits[]` (`from`/`to`/`reason`), `meta.llm_model`,
  `meta.postproc_version`;
- `meta.version = "baseline-0.2"`, `confidence = 0.0` (честно «не измерено»),
  сегменты упорядочены по `start`.

Выборка 3 edits (все `reason="punct"`):

1. `"расскажи" → "Расскажи"` — капитализация начала предложения; осмысленно.
2. `"маршруте" → "маршруте."` — финальная точка; осмысленно.
3. `"дел пожалуйста" → "дел. Пожалуйста,"` — вставка точки + капитализация +
   запятая; осмысленно, но правка захватывает два слова, а `reason` всё равно
   `"punct"` (задокументированный fallback).

Замечание: сегмент 6 начинается со строчной «на каком этапе…» — следствие
выравнивания LLM-текста по ASR-сегментам, контракт не нарушен, но отдельный
сегмент не всегда является самостоятельным предложением.

## 3. Воспроизводимость

Повторный merge v0.2 в `D:\temp\opencode\review_llm_merge` с теми же входами
(`--llm-hyp reports\postproc_eval\llm_torch_correct13_punct.jsonl`,
`--llm-model "Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning int4"`,
`--postproc-version "postproc_v1 (punctuate+correct)"`):

- 13/13 valid, 84 сегмента, 44 сегмента с edits, 63 правки, 940 символов,
  0 doubtful — совпадает с `REPORT_LLM.md`;
- SHA256 по 13 файлам vs `output/baseline_llm`: **0 отличий**.

Повторный merge v0.1 (без `--llm-*`) в `D:\temp\opencode\review_llm_v01`:

- 13/13 valid, 84 сегмента, 0 edits;
- SHA256 по 13 файлам vs `output/baseline`: **0 отличий** (регресс v0.1
  подтверждён).

Повторный `eval_postproc` на `llm_torch_punct300.jsonl`:

- метрики идентичны (P/R/F1 0.817875/0.703919/0.756630, boundary F1 0.848580);
- SHA256 отчёта `punct300.json` vs `punct300_2.json`: **равны** (отчёт
  детерминирован).

## 4. Лицензии

| Проверка | Результат |
|---|---|
| `models/llm/LICENSE` | Apache License, Version 2.0 (строка 2) — присутствует |
| `models/llm/NOTICE` | Apache-2.0, атрибуция QVikhr, ревизия `ff7a23e…`, база Qwen3-1.7B (Apache-2.0) — присутствует |
| `models/llm/MANIFEST.md` | лицензия Apache-2.0, ревизия, sha256 исходника — указаны |
| `data/postproc/build_stats.json` | `punct_source = Tatoeba Russian sentences`, `punct_license = CC-BY-2.0-FR` — присутствует |
| `NOTICE` (evaluation data) | Tatoeba CC-BY 2.0 FR + correction CC-BY-4.0/MIT — указаны |

Несостыковки:

- Лицензия обучающего датасета `Vikhrmodels/GrandMaster2` **не проверена**
  (нет карточки/файла в репозитории). Это явно признано в `MANIFEST.md` §6 и
  `POSTPROC_REPORT.md` §6; для Apache-2.0 весов не блокер, но требует
  лицензионного аудита.
- Форматирование `CC-BY-2.0-FR` (build_stats) vs `CC-BY 2.0 FR` (NOTICE) —
  одна и та же лицензия, косметическое расхождение.

## 5. Честность оценок для мобилы (`POSTPROC_REPORT.md` §3–4)

### 5.1 Что сходится

- **KV-кэш:** 28 слоёв × 2 (K/V) × 8 KV-heads × 128 head_dim × 2 байта =
  114 688 Б = 0.1147 MB (десятичных) = 0.1094 MiB на токен. Контекст
  256/512/1024 → 29.36/58.72/117.44 MB — совпадает с отчётом.
- **Пик RAM LLM:** 1.108 + 0.0294 + 0.15 = 1.287 ГБ; 1.108 + 0.1174 + 0.15 =
  1.375 ГБ → «1.29–1.38 ГБ» — сходится.
- **Декод (bandwidth-bound):** 25 × 0.6 / 1.108 = 13.54 ток/с;
  51 × 0.75 / 1.108 = 34.52 ток/с → «13.5 / 34.5» — сходится.
- **Сегментов на минуту:** 310.641 / 84 = 3.698 с/сегмент; 60 / 3.698 = 16.22
  → «16.2» — сходится.
- **`punctuate`, с/сегмент:** 83.9/50 + 22.7/10 = 3.95; 83.9/100 + 22.7/15 =
  2.35; 83.9/150 + 22.7/25 = 1.47 → совпадает с таблицей.
- **`punctuate`, с/мин:** 3.95/2.35/1.47 × 16.2 = 64.0/38.1/23.8 → «64.1/38.2/23.8»
  — сходится.
- **ASR:** 0.018949 × 5…10 = 0.0947…0.1895; 60 × RTF = 5.68…11.37 с/мин →
  «0.095–0.19; 5.7–11.4» — сходится.
- **Диаризация:** 0.0775 × 5…10 = 0.3875…0.775; 60 × RTF = 23.25…46.5 с/мин →
  «0.39–0.78; 23–47» — сходится.
- **Итог пайплайна:** min 0.2+5.7+23.3+24 = 53.2; max 0.5+11.4+46.5+64 = 122.4;
  среднее 0.35+8.55+34.9+38 = 81.8 → «53–122, ~82»; RTF 0.88–2.04 → «0.9–2.0»
  — сходится. Full: 79.2…118.4, среднее 98.8 → «79–118, ~99» — сходится.

### 5.2 Где арифметика/пометки расходятся

- **Вес «1.108 ГБ» приписан не тем файлам.** `model.onnx` + `model.onnx.data`
  = 1 097 039 650 Б = **1.097 ГБ**, а 1.108 ГБ — весь каталог (с tokenizer и
  конфигами, `MANIFEST.md` §3). Влияние на пик RAM ~11 МБ, выводы не меняются.
- **`full`, «~650 ток/мин» (§3) не выводится из замеров.** Измерено 5775
  generated-токенов на 63 элемента = 91.7/элемент. При 16.2 сегм./мин это
  ~1485 ток/мин → ~67 с/мин, а не 29. Число 650 соответствует ~7 вызовам/мин;
  допущение о батчинге/абзацах не указано.
- **Строки `correct`/`full` в §4.3 не воспроизводятся по приведённой формуле
  при 16.2 сегм./мин.** По формуле и токенам получается `correct` ~84–223 с/мин,
  `full` ~97–262 с/мин, а в таблице 15–40 и 35–85. Табличные значения
  соответствуют ~3 вызовам/мин (`correct`) и ~5–6 (`full`), т.е. неявному
  батчингу по записи/абзацу; это допущение не задокументировано.
- **Единицы KV-кэша:** «МБ» посчитаны как десятичные MB (0.1147 MB), а не MiB
  (0.1094 MiB); в тексте смешаны MB/MiB/GB. На выводы не влияет.
- **§4.8, последовательный пик:** max(LLM 1300–1500) + приложение 100–200 =
  1400–1700 МБ, а заявлено 1500–1800 МБ — небольшое завышение.

### 5.3 Оценка vs замер

Раздел §4 явно озаглавлен «Оценка для мобилы (аналитика, не замер)», в §4.7
есть колонка «Статус» с пометкой «оценка», §3 — GPU-замеры. Оценки за замеры
в целом **не выдаются**. Исключения — пункты 5.2: «1.108 ГБ весов» (измерение
не тех файлов) и необъяснённые «650 ток/мин» / строки `correct`/`full` в §4.3,
где не показано допущение о батчинге.

## 6. Вердикт

**PASS WITH NOTES.** Критичных замечаний — 0, некритичных — 9.

### Замечания

1. *(некритично)* `POSTPROC_REPORT.md` §4.1: «1.108 ГБ по файлам
   `model.onnx` + `model.onnx.data`» — фактически 1.097 ГБ; 1.108 ГБ — весь
   каталог. Влияние ~11 МБ.
2. *(некритично)* `POSTPROC_REPORT.md` §3: «~650 ток/мин» для `full` не
   выводится из измеренных 5775 generated / 63 = 91.7 на элемент и 16.2
   сегм./мин (получается ~1485 ток/мин → ~67 с/мин). Нужно указать допущение
   о батчинге.
3. *(некритично)* `POSTPROC_REPORT.md` §4.3: строки `correct`/`full` не
   воспроизводятся по приведённой формуле при 16.2 сегм./мин; неявно
   предполагается ~3 и ~5–6 вызовов/мин. Указать допущение.
4. *(некритично)* `POSTPROC_REPORT.md` §4.1: KV-кэш в «МБ» посчитан в
   десятичных MB (0.1147 MB = 0.1094 MiB); смешение MB/MiB/GB.
5. *(некритично)* `POSTPROC_REPORT.md` §4.8: последовательный пик 1500–1800 МБ
   vs max(LLM 1300–1500) + приложение 100–200 = 1400–1700 МБ.
6. *(некритично)* `output/baseline_llm/*.json`: `reason="punct"` для всех 63
   правок, включая 9 словарных замен и multi-word replace
   (`"дел пожалуйста" → "дел. Пожалуйста,"`). Задокументировано в
   `REPORT_LLM.md` §6.3, но снижает семантическую ценность `edits`.
7. *(некритично)* `output/baseline_llm/diar_0001.json`: сегмент 6 начинается
   со строчной «на каком…» — следствие выравнивания LLM-текста по
   ASR-сегментам; контракт не нарушен, но сегмент не всегда самостоятелен.
8. *(некритично)* Лицензия обучающего датасета `Vikhrmodels/GrandMaster2` не
   проверена (признано в `MANIFEST.md` §6 и `POSTPROC_REPORT.md` §6); нужен
   лицензионный аудит.
9. *(некритично)* ONNX int4 проверен только на 10 примерах (exact 3/10,
   char-sim 0.9856, punct F1 −0.08) — не полноценный quality gate (признано).

### Не проверено (и почему)

- Реальный замер RTF/RAM на Android — нет устройства; в отчёте это честно
  помечено как оценка.
- Лицензия `GrandMaster2` — нет карточки/файла в репозитории.
- Повторный GPU-замер скорости — LLM-генерация запрещена заданием; сверены
  только сохранённые `*_metrics.json` (значения совпадают с отчётом).
- ONNX int4 на наборе больше 10 примеров — готовых артефактов нет.

## 7. Артефакты проверки

- Отчёты eval: `D:\temp\opencode\review_llm\{punct300,correct13,full63}.json`
  (+ `.txt`), повтор `punct300_2.json`.
- Повторный merge v0.2: `D:\temp\opencode\review_llm_merge\` (0 отличий SHA256).
- Повторный merge v0.1: `D:\temp\opencode\review_llm_v01\` (0 отличий SHA256).
- Журнал: `logs/journal.md` (дописана строка reviewer).
