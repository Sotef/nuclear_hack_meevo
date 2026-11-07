# T-Wix: источник, лицензия, схема и подготовка

Дата: 2026-11-07. Агент: data-eval.

## 1. Источник и ревизия

- **URL:** https://huggingface.co/datasets/t-tech/T-Wix
- **Ревизия (commit sha):** `0059b0967915624069b5b4dc224ebe7ef7c041a6`
- **Автор:** `t-tech`
- **Название:** T-Wix SFT Mixture
- **Языки:** ru, en (тег `language:ru`, `language:en`)
- **Размер:** 499 598 строк; 8 parquet-файлов; download 1 701 308 457 байт (~1.70 ГБ);
  dataset_size 3 628 841 342 байта (~3.63 ГБ).
- **Локальная копия:** `data/twix/raw/` (8 parquet + README.md + .gitattributes).

### Кандидаты с похожим именем

При поиске на HF найдены:
- `t-tech/T-Wix` — **канонический** русский SFT-датасет (выбран).
- `concretejungles/T-Wix-instag` — явно указан как derivative of `t-tech/T-Wix`
  (добавлена колонка `supertag`).
- `concretejungles/T-Wix-instag-graded` — дальнейший derivative предыдущего
  (добавлена колонка `reward`).
- `bethrezen/twix-think` — отдельный датасет с похожим именем, не заявлен как
  derivative; не использовался.
- `Nebliss/*Twix*`, `Hutti09/*Twix*`, `swp4065/grab_twix`, `MaxHi/gm_twix_*` —
  **нерелевантные robotics/LeRobot-датасеты** (манипулятор Twix), не текст.

Канонический T-Wix однозначно определяется как `t-tech/T-Wix`; производные
ссылаются на него явно. Если для задачи нужен именно derivative (например,
с reward-разметкой), это отдельное решение пользователя.

## 2. Лицензия

- **Лицензия:** **ODC-BY-1.0** (Open Data Commons Attribution License 1.0).
  Указана в README датасета; HF-тег — `license:odc-by`.
- README прямо оговаривает: датасет включает выходы сторонних моделей
  (DeepSeek-V3-0324, Qwen3-235B-A22B и др.), которые могут иметь отдельные
  условия; «See each subset link for specific licensing details».
- Для использования в приложении: ODC-BY требует атрибуции; коммерческое
  использование разрешено. Атрибуцию и оговорку про third-party outputs нужно
  сохранить в notices. Точная формулировка README:
  «This dataset is licensed under ODC-BY-1.0. It includes outputs from
  third-party models, which may be subject to separate terms.»
- Отдельного файла `LICENSE` в репозитории нет — только README.

## 3. Схема полей

Parquet-схема (pyarrow):

```
id: string
messages: list<element: struct<content: string, role: string>>
subset: string
```

- `id` — уникальный id примера (UUID-подобная строка).
- `messages` — список сообщений; `role` ∈ {system, user, assistant};
  `content` — текст сообщения.
- `subset` — `general`, `reasoning`, `long_context`, `english_corpus`.

Распределение по subset (посчитано по всем 499 598 строкам):

| subset | строк |
|---|---:|
| general | 427 843 |
| reasoning | 30 984 |
| long_context | 4 363 |
| english_corpus | 36 408 |
| **итого** | **499 598** |

Распределение числа сообщений: 2 сообщения — 462 890; 3 — 13 040; 4 — 3 277;
5 — 1 338; 6 — 9 319; далее длинные диалоги до 42 сообщений.
Длины (символы): user — median 351, mean 923, p95 1 876, max 255 819;
assistant — median 2 877, mean 3 540, p95 7 352, max 100 460.

### Примеры строк (5 из 8, `data/twix/samples.json`)

1. `id=4ac02c28-c224-4231-aadf-8d9775eb4718`, subset=`general`, roles=[user, assistant]
   - user: «Вычислите преобразование Фурье гауссовой функции e^(-x^2).»
   - assistant: «Чтобы вычислить преобразование Фурье гауссовой функции \( f(x) = e^{-x^2} \), воспользуемся определением…» (математика, LaTeX)
2. `id=8e1d55e8-bb3a-4c62-8d66-9a32652bae10`, subset=`general`
   - user: «Вы работаете с приложением на Ember.js и вам нужно создать тест для компонента "nav-bar"…»
   - assistant: пример теста на Ember-QUnit/HTMLBars (код).
3. `id=6151b39c-7f7e-401a-8fab-b26f64da3891`, subset=`general`
   - user: «…реализовать класс Space-Partitioning Tree (SPTree) на C++…»
   - assistant: реализация класса на C++.
4. `id=4f930ade-d327-47b9-a966-ea2153f2edcf`, subset=`general`
   - user: «…задача, сосредоточенная на реализации структур данных и алгоритмов на языке C…»
   - assistant: структура проекта и код на C.
5. `id=afbd74eb-1794-4c0d-be2f-9100d2c8df15`, subset=`reasoning`
   - user: вопрос по уравнению Линдблада с вариантами A–D и `/think`.
   - assistant: ` thinking…` + рассуждение (reasoning-трейс).

Полные примеры — `data/twix/samples.json`; примеры по ключевым словам
(пунктуация/исправление/опечатка) — `data/twix/keyword_samples.json`.

## 4. Что подготовлено

| Файл | Строк | Описание |
|---|---:|---|
| `data/twix/train_smoke.jsonl` | **10 000** | Детерминированная выборка, `seed=42`: `random.Random(42).sample(range(499598), 10000)`, отсортировано по глобальному индексу строки. Формат — исходные записи `{id, messages, subset}`. |
| `data/twix/train_full.jsonl` | **499 598** | Полный датасет в исходном порядке файлов, те же поля. |
| `data/twix/stats.json` | — | Строки, поля, длины, subset-распределение, источник, ревизия, лицензия, seed, метод отбора. |
| `data/twix/samples.json` | 8 | Примеры строк для отчёта. |
| `data/twix/keyword_samples.json` | 40 | Примеры user-промптов с ключевыми словами (пунктуация/исправление/опечатка). |

Проверка: `train_smoke.jsonl` — ровно 10 000 строк; `train_full.jsonl` — 499 598
строк; размеры 75.6 МБ и 3.72 ГБ соответственно.

## 5. Подходит ли T-Wix под пунктуацию/коррекцию

**Нет, напрямую не подходит.** T-Wix — это общий русский SFT-mixture
(инструкции, математика, код, reasoning, long context, english corpus).
В нём **нет** пар «сырой ASR → исправленный текст» и **нет** пар
«текст без пунктуации → текст с пунктуацией». Поля `messages` содержат
обычные user/assistant-диалоги, а не пары постпроцессинга.

Поэтому **обучающие пары под задачи постпроцессора не выдумывались**.
`train_smoke.jsonl` / `train_full.jsonl` — это исходные SFT-записи, а не
postproc-пары. Использовать их можно как:
- вспомогательный SFT для русского языка/инструкций (general capability);
- источник «чистых» assistant-текстов, если позже будет согласована
  синтетическая схема порчи текста (например, удаление пунктуации и/или
  внесение ASR-подобных ошибок) — но это отдельная деривация, требует
  подтверждения и не входит в текущее задание;
- для few-shot/стилевых примеров, но не как готовый датасет пунктуации.

Для собственно постпроцессора нужен отдельный датасет с парами
`input (ASR/no-punct) → output (ref)`. В проекте уже есть:
`data/postproc/punct_eval.jsonl` (300 пар Tatoeba) и
`data/postproc/correction_eval.jsonl` (13 пар ASR→ref); их можно использовать
как eval, а для обучения — расширять/собирать отдельно.

## 6. Ограничения

- Домен общий (инструкции/код/математика), не спонтанная речь и не ASR-канал.
- `english_corpus` (36 408 строк) — английский, не русский; при обучении
  русскому постпроцессору его, вероятно, нужно исключать.
- `long_context` содержит очень длинные тексты (user до 255 819 символов);
  для обучения с ограниченным контекстом нужна нарезка/фильтрация.
- Лицензия ODC-BY-1.0 требует атрибуции; third-party model outputs могут иметь
  отдельные условия — юридический аудит перед продуктом.
- Датасет может содержать шум/биасы (явно указано в README).
- `train_full.jsonl` — 3.72 ГБ; при нехватке диска можно пересобрать из
  `data/twix/raw/` тем же скриптом (метод зафиксирован в `stats.json`).
