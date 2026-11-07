# Спецификация eval-харнесса LLM-постпроцессора

Инструмент: `tools/eval_postproc/eval_postproc.py` 1.0.0
Дата: 2026-11-06
Python: 3.12.10 (`.venv\Scripts\python.exe`)

## 1. Наборы данных

### 1.1 `data/postproc/punct_eval.jsonl` — восстановление пунктуации

| Поле | Описание |
|---|---|
| `id` | `punct_0001` … `punct_0300` |
| `text_raw` | Текст без пунктуации/разметки (слова через пробел, нижний регистр — как выход ASR) |
| `text_ref` | Референс с пунктуацией |
| `source` | `Tatoeba Russian sentences (rus_sentences.tsv.bz2)` |
| `license` | `CC-BY-2.0-FR` |
| `tatoeba_id` | Исходный id предложения Tatoeba |
| `clean` | `true` для 50 предложений, используемых как «чистый текст» (false-alarm proxy) |

- 300 пар, отобраны детерминированно (`seed=42`) из 315 600 подходящих
  предложений русского экспорта Tatoeba.
- Фильтры: только кириллица/пробелы/допустимая пунктуация, 5–30 слов,
  длина ≤ 300 символов, минимум 2 знака пунктуации, финальный `.`/`!`/`?`,
  дедупликация по `text_raw`.
- Лицензия: Tatoeba для текстовых предложений по умолчанию применяет
  **CC-BY 2.0 FR** (атрибуция обязательна, коммерческое использование разрешено).
  Отдельные предложения могут иметь другую CC-лицензию; экспорт не содержит
  per-sentence лицензию, поэтому записана консервативная лицензия по умолчанию
  `CC-BY-2.0-FR`. Атрибуция: Tatoeba contributors, https://tatoeba.org.
- 50 «чистых» предложений — это первые 50 по `id` после детерминированной
  выборки; они не содержат ASR-ошибок по построению (это референсные тексты).

### 1.2 `data/postproc/correction_eval.jsonl` — исправление ошибок ASR

| Поле | Описание |
|---|---|
| `id` | id записи из `data/test` (`ruls_0001` … `diar_0003`) |
| `text_asr` | Гипотеза ASR (`tools/eval_asr/hyp/small_zipformer_ru/<id>.txt`) |
| `text_ref` | Эталон (`data/test/<id>.txt`, без пунктуации, нижний регистр) |
| `source` | `data/test reference + tools/eval_asr/hyp/small_zipformer_ru` |
| `license` | `CC-BY-4.0 / MIT` (по `data/test/manifest.jsonl`) |

- 13 пар (по одной на запись). Baseline WER ASR = **0.146881** (497 слов),
  совпадает с `eval_asr` 1.0.0.
- Пары не разбиты по фразам: в эталонах нет пунктуации, поэтому границы фраз
  не определены. Это ограничение набора.

## 2. Формат входа `--hyp`

JSONL, одна строка — один объект:

```json
{"id": "punct_0001", "text": "Один, два, три.", "doubtful": ["три"], "edits": [{"from": "три", "to": "два", "reason": "ASR"}]}
```

- `id` — обязателен, должен совпадать с id из `punct_eval.jsonl` или
  `correction_eval.jsonl`.
- `text` — обязателен, текст после LLM.
- `doubtful` — необязательный список строк или объектов. Строка — сомнительный
  фрагмент. Объект может содержать `span`/`text`/`from` и `suggestions`.
- `edits` — необязательный список `{from, to, reason}`.
- Допускается inline-разметка внутри `text`:
  `[doubtful: слово]` или `[doubtful: слово | вариант1 | вариант2]`.
  Маркеры удаляются перед оценкой пунктуации/WER.
- Альтернатива: `--hyp` может быть каталогом `<id>.txt` (только inline-разметка).

## 3. Метрики

### 3.1 Пунктуация

Текст разбивается на слова; знаки из набора
`{., !, ?, ,, ;, :, —, –, …}` приписываются предшествующему слову.
Сравнение — по выровненным словам (`difflib.SequenceMatcher` по
нормализованным словам: lower, `ё→е`).

Для каждой пары слов:

- `TP` — знаки, совпавшие в ref и hyp;
- `FP` — знаки в hyp, которых нет в ref;
- `FN` — знаки в ref, которых нет в hyp.

Для несовпавших участков: знаки ref → `FN`, знаки hyp → `FP`.

```
precision = TP / (TP + FP)      (1.0, если TP+FP=0)
recall    = TP / (TP + FN)      (1.0, если TP+FN=0)
F1        = 2 * precision * recall / (precision + recall)
```

- **micro** — суммы TP/FP/FN по всем предложениям, затем формулы.
- **macro** — среднее per-sentence P/R/F1.

Границы предложений: слово считается границей, если после него есть
`.`/`!`/`?`/`…`.

```
boundary_precision = TP_b / hyp_boundaries
boundary_recall    = TP_b / ref_boundaries
boundary_f1        = 2 * P_b * R_b / (P_b + R_b)
```

`boundary_recall` — «доля корректно восстановленных границ предложений».

### 3.2 Коррекция (WER)

Нормализация как в `eval_asr` 1.0.0: lower, `ё→е`, удаление всего кроме
`0-9a-zа-я` и пробелов, схлопывание пробелов.

```
WER(text_hyp, text_ref) = edit_distance(ref_words, hyp_words) / len(ref_words)
WER_before = WER(text_asr, text_ref)
WER_after  = WER(text_hyp, text_ref)
delta_WER  = WER_after - WER_before
```

- `delta_WER < 0` — улучшение, `> 0` — ухудшение, `= 0` — без изменений.
- micro WER — сумма ошибок / сумма слов референса по всем парам.
- macro WER — среднее per-pair WER.
- `improved` / `worsened` / `unchanged` — число пар по знаку `delta_WER`
  (допуск `1e-12`).

### 3.3 Флаги «сомнительное словосочетание»

**False-alarm proxy.** На 50 предложениях с `clean=true`:

```
false_alarm_rate = (# clean-предложений с >=1 doubtful-флагом) / (# clean-предложений в hyp)
flag_density_per_100_words = 100 * (# doubtful-флагов) / (# слов в clean-текстах)
```

Учитываются только флаги `doubtful` (inline + поле `doubtful`); `edits` —
это исправления, а не флаги.

**Покрытие известных ASR-ошибок.** Для каждой пары `correction_eval`:

1. Выравниваются слова `text_asr` и `text_ref`.
2. Множество ошибочных лексем = слова из замен/удалений/вставок
   (surface form после нормализации).
3. Флаг покрывает лексему, если нормализованные слова из `doubtful`/`edits.from`
   пересекаются с множеством ошибочных лексем.

```
error_coverage = (# покрытых ошибочных лексем) / (# всех ошибочных лексем)
pair_coverage  = (# пар, где покрыта >=1 ошибка) / (# пар с ошибками)
```

Ограничение: покрытие считается по surface form, без лемматизации и без
позиционного сопоставления флага с конкретным словом в ASR-гипотезе.

## 4. Выход

`--report <path>`: если путь оканчивается на `.json`, пишется он и sibling
`.txt`; иначе добавляются `.json` и `.txt`. Без `--force` существующие файлы
не перезаписываются (код возврата 2).

`report.json` содержит:

- `tool`, `version`, `dependencies` (Python, сторонних зависимостей нет);
- `inputs` (пути, число записей, missing/unknown id);
- `punctuation.summary` + `punctuation.per_sentence`;
- `correction.summary` + `correction.per_pair`;
- `doubtful` (false-alarm, плотность флагов, покрытие ошибок).

## 5. Ограничения набора

- **Домен**: Tatoeba — бытовые письменные предложения, не спонтанная речь и не
  телефонный канал. Пунктуация в них авторская, а не результат разметки ASR.
- **Лицензия**: CC-BY-2.0-FR требует атрибуции Tatoeba; для приложения это
  допустимо, но атрибуцию нужно сохранить в notices.
- **Пунктуация ASR-эталонов**: в `data/test/*.txt` пунктуации нет, поэтому
  `correction_eval` не оценивает пунктуацию; для неё используется отдельный
  `punct_eval`.
- **Размер correction-набора**: 13 записей, 497 слов; «доля улучшенных пар»
  статистически грубая (шаг 1/13).
- **Флаги**: false-alarm proxy измеряется на чистых референсных предложениях,
  а не на реальных ASR-гипотезах; покрытие ошибок — приближение по surface form.
- **Знаки**: кавычки, скобки, дефисы внутри слов не входят в метрику пунктуации;
  `…` считается отдельным знаком, границей предложения — да.
- **Регистр**: пунктуационная метрика регистронезависима; `text_raw` приведён
  к нижнему регистру (как выход ASR), `text_ref` сохраняет исходный регистр.

## 6. Воспроизведение

```powershell
# 1. Сборка наборов (детерминированно, seed=42)
.venv\Scripts\python.exe data\postproc\build_postproc_data.py

# 2. Self-test: hyp = ref для пунктуации, hyp = ASR для коррекции
.venv\Scripts\python.exe reports\postproc_eval\make_selftest.py
.venv\Scripts\python.exe tools\eval_postproc\eval_postproc.py `
  --hyp reports\postproc_eval\selftest_ref.jsonl `
  --ref-punct data\postproc\punct_eval.jsonl `
  --ref-corr data\postproc\correction_eval.jsonl `
  --report reports\postproc_eval\selftest_ref.json --force

# 3. Self-test с удалёнными запятыми (F1 должен упасть)
.venv\Scripts\python.exe tools\eval_postproc\eval_postproc.py `
  --hyp reports\postproc_eval\selftest_no_commas.jsonl `
  --ref-punct data\postproc\punct_eval.jsonl `
  --ref-corr data\postproc\correction_eval.jsonl `
  --report reports\postproc_eval\selftest_no_commas.json --force

# 4. Self-test флагов
.venv\Scripts\python.exe tools\eval_postproc\eval_postproc.py `
  --hyp reports\postproc_eval\selftest_flags.jsonl `
  --ref-punct data\postproc\punct_eval.jsonl `
  --ref-corr data\postproc\correction_eval.jsonl `
  --report reports\postproc_eval\selftest_flags.json --force
```

## 7. API-контракт ML-модуля

Краткая заметка: `docs/ml_api_contract.md`. Базовый `OUTPUT CONTRACT` не
меняется; перечислены только lifecycle-методы, поля запроса, additive-поля
вывода, коды ошибок и стадии прогресса.
