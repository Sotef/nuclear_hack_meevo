# RUPunct-small ONNX — замер на punct_eval (300 пар)

Дата: 2026-11-07. Автор: asr-engineer.
Модель: `ekhodzitsky/rupunct-small-onnx` (INT8, база `RUPunct/RUPunct_small`).
Метрики: `tools/eval_postproc` 1.0.0, Python 3.12.10, `.venv`.
Набор: `data/postproc/punct_eval.jsonl` (300 пар, Tatoeba, CC-BY-2.0-FR).
Детерминизм: повторный прогон даёт идентичный SHA-256 гипотез (`88164cc7…`).

## 1. Таблица: RUPunct-small vs QVikhr

| Метрика | RUPunct-small INT8 | QVikhr-3-1.7B int4 (baseline) |
|---|---:|---:|
| Punct P (micro) | **0.860643** | 0.817875 |
| Punct R (micro) | **0.815675** | 0.703919 |
| Punct F1 (micro) | **0.837556** | 0.756630 |
| Boundary F1 | **0.947853** | 0.848580 |
| Boundary P / R | 0.974763 / 0.922388 | — |
| TP / FP / FN | 562 / 91 / 127 | — |
| Скорость (CPU, PC) | **2.14 мс/сегмент** (300 за 0.642 с, inference) | ~12.8 с/сегмент на устройстве* |
| Загрузка модели | 4.9 с (tokenizer + ORT session) | — |
| Размер | **27.9 MiB** ONNX (+2.3 MB tokenizer) | 1.10 GB int4 |
| Лицензия | **MIT** | Apache-2.0 |
| Язык | ru-only | multilingual/ru |

\* QVikhr — wall-clock на SM-N9750 из `logs/journal.md` (2026-11-07); GPU-замер
(`COMPARE.md`) посегментную скорость не фиксировал. Прямое сравнение скорости
некорректно (PC CPU vs Android), но разрыв ~6000× по latency показателен.

Порог замены: **F1 ≥ 0.72 и boundary ≥ 0.80** → RUPunct проходит с запасом
(0.8376 ≥ 0.72, 0.9479 ≥ 0.80).

## 2. RU-примеры (raw → RUPunct → ref)

| id | text_raw | RUPunct hyp | text_ref |
|---|---|---|---|
| punct_0001 | добрый вечер как ваши дела | Добрый вечер, как ваши дела! | Добрый вечер, как ваши дела? |
| punct_0002 | грубо говоря ты должен исходить из расчёта фунт говядины на двух гостей | Грубо говоря, ты должен исходить из расчёта фунт говядины на двух гостей. | Грубо говоря, ты должен исходить из расчёта фунт говядины на двух гостей. |
| punct_0003 | сказав я скучала по тебе она начала плакать | Сказав я— скучала по тебе, она начала плакать. | Сказав «я скучала по тебе», она начала плакать. |
| punct_0004 | ты в узкой скальной шахте далеко вверху брезжит свет под тобой пещера в которой всё началось что ты будешь сейчас делать | Ты в узкой скальной шахте далеко вверху брезжит свет под тобой пещера, в которой всё началось, что ты будешь сейчас делать. | Ты в узкой скальной шахте. Далеко вверху брезжит свет. Под тобой пещера, в которой всё началось. Что ты будешь сейчас делать? |
| punct_0005 | я хорошо плавал когда был ребёнком | Я хорошо плавал, когда был ребёнком. | Я хорошо плавал, когда был ребёнком. |

Кириллица работает корректно. **Склейки/обрезки слов нет**: число слов
(raw vs hyp) совпало на 300/300. Ограничения: не ставит кавычки «», иногда
путает тире/запятую и не всегда делит длинное предложение на несколько.

## 3. Точный формат модели (для порта в Kotlin)

**Tokenizer** — HF WordPiece `BertTokenizer` (`tokenizer.json`):
- `do_lower_case=false`, `strip_accents=false`, `add_prefix_space=true`;
- normalizer `BertNormalizer` (clean_text, handle_chinese_chars, lowercase=false);
- pre-tokenizer `BertPreTokenizer` (разбиение по пробелам/пунктуации);
- vocab 83 828, `[PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 [MASK]=4`;
- `max_position_embeddings=2048` (в замере truncation=512).

**Вход ONNX** (int64, `[1, T]`): `input_ids`, `attention_mask`, `token_type_ids`.
**Выход**: `logits` float32 `[1, T, 33]`.

**33 метки** = 3 режима регистра × 11 классов пунктуации:
- префиксы: `LOWER_` (как есть), `UPPER_` (Capitalize), `UPPER_TOTAL_` (UPPER);
- классы: `O` (""), `PERIOD` (.), `COMMA` (,), `QUESTION` (?), `TIRE` (—),
  `DVOETOCHIE` (:), `VOSKL` (!), `PERIODCOMMA` (;), `DEFIS` (-),
  `MNOGOTOCHIE` (...), `QUESTIONVOSKL` (?!).

**Алгоритм декода** (воспроизводит upstream `process_token`):
1. argmax по 33 логитам на каждый токен;
2. aggregation `first`: для каждого слова берётся метка **первого** субтокена;
   слово восстанавливается по полному span (offset первого…последнего субтокена);
3. `process_token(word, label)`: применить регистр (lower/capitalize/upper) и
   дописать пунктуацию; для `TIRE` в не-lower режиме — `word + " —"`;
4. слова соединить одним пробелом.

Kotlin: ONNX Runtime Mobile + WordPiece-токенизатор (BertPreTokenizer + WordPiece,
нужны `word_ids`/offsets для группировки субтокенов). Модель stateless, батч=1.

## 4. Вердикт

**PASS — рекомендуется заменить QVikhr-punctuate на RUPunct-small INT8.**
F1 0.8376 (порог 0.72), boundary F1 0.9479 (порог 0.80), размер 27.9 MiB,
MIT, 2.14 мс/сегмент на CPU. Оговорки: модель делает только пунктуацию+регистр
(без коррекции ASR и без doubtful-флагов), не ставит кавычки; для задачи
`punctuate` это прямой апгрейд baseline.

## 5. Артефакты

- Гипотезы: `reports/postproc_eval/rupunct_punct300.jsonl`
- Отчёт eval: `reports/postproc_eval/rupunct_punct300.json` / `.txt`
- Тайминг: `reports/postproc_eval/rupunct_punct300_timing.json`
- Раннер: `reports/asr_engineer/run_rupunct.py`
- Манифест модели: `models/punct_ru/MANIFEST.md`
