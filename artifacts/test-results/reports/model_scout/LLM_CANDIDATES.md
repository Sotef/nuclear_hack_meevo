# LLM-кандидаты для офлайн-постобработки транскрипций (RU, Android arm64)

Дата: 2026-11-06. Автор: model-scout.
Статус: разведка по карточкам/докам (webfetch). Веса не скачивались, инференс не запускался.
Смежные отчёты: `reports/model_scout/CANDIDATES.md` (ASR/VAD/диаризация) — здесь не дублируются.

Задача LLM в пайплайне: пунктуация/регистр, исправление ошибок ASR, флаг «сомнительное
словосочетание» + варианты замены. Вход/выход — короткие реплики (десятки–сотни токенов),
поэтому большое контекстное окно не критично; критичны размер, скорость и качество русского.

HARD CONSTRAINTS (напоминание): Android arm64-v8a; разрешены ONNX Runtime Mobile,
sherpa-onnx, whisper.cpp, TFLite/LiteRT. llama.cpp/GGUF, MLC и PyTorch в приложении —
**не разрешены**; использование GGUF = эскалация пользователю на изменение констрейнтов.
sherpa-onnx и whisper.cpp — ASR-рантаймы, LLM в них не запускается; фактически для LLM
остаются **ONNX Runtime (GenAI/Mobile)** и **LiteRT/MediaPipe LLM Inference**.

Обозначения:
- **«не указано»** — нет в проверенном источнике.
- **«оценка»** — расчёт из числа параметров (BF16 ≈ 2 байта/параметр, int8 ≈ 1 байт,
  int4 ≈ 0.5 байта + накладные расходы), не измерение.
- Даты — по карточке HF / arXiv; ревизии (commit sha) не фиксировались, т.к. веса не качались.

---

## 0. Краткий вывод

| Кандидат | Вердикт |
|---|---|
| **QVikhr-3-1.7B-Instruction-noreasoning** | Рабочий кандидат. Apache-2.0, сильный RU, 1.7B. Нет официального ONNX → нужен экспорт через ONNX Runtime GenAI model builder; риск средний. |
| **Gamayun 1.5B** | **Не кандидат для продукта сейчас**: веса и лицензия не опубликованы (только статья arXiv 2512.21580). Даже при публикации — LLaMA-like, экспорт возможен, но артефактов нет. |
| Фолбэк 1: **Qwen2.5-1.5B-Instruct** | Apache-2.0, официальная поддержка Qwen2/Qwen2.5 в ONNX Runtime GenAI, предсказуемый экспорт. RU чуть слабее QVikhr. |
| Фолбэк 2: **Phi-3.5-mini-instruct** | MIT, 3.8B, 128k контекст. Тяжёлый для среднего телефона, RU-качество не подтверждено бенчмарками. |

---

## 1. Кандидат 1 — QVikhr-3-1.7B-Instruction-noreasoning

### 1.1. Карточка

| Поле | Значение |
|---|---|
| Точное имя | `Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning` (отдельного репозитория `QVikhr-3-1.7B-Instruction` без суффикса `-noreasoning` в поиске HF не найдено; для 4B есть `QVikhr-3-4B-Instruction`) |
| URL | https://huggingface.co/Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning |
| Владелец | Vikhrmodels (Vikhr Team; авторы: S. Bratchikov, N. Kompanets, K. Korolev, A. Nikolich) |
| Базовая модель | `Qwen/Qwen3-1.7B` (дерево: `Qwen/Qwen3-1.7B-Base` → `Qwen/Qwen3-1.7B` → QVikhr) |
| Версия / дата | Обновление репозитория/карточки: **30.05.2025**; отдельного semver-тега нет. Дата создания в карточке не указана. |
| Размер весов | 1.7B параметров (HF показывает «2B params», BF16). BF16 ≈ **3.4 ГБ**; int8 ≈ **1.7–1.8 ГБ**; int4 ≈ **0.9–1.1 ГБ** (оценка; ориентир — GGUF Q4_K_M ≈ 1.1 ГБ). |
| Квантизации, которые реально есть | Официальные: **GGUF** (`Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning-GGUF`), **MLX 4bit/8bit**. Официального int8/int4 ONNX нет. Community GGUF: `mradermacher/...-i1-GGUF`, `DevQuasar/...-GGUF`. |
| Контекстное окно | В карточке QVikhr **не указано**. У базовой Qwen3-1.7B — 32 768 токенов (131 072 с YaRN). Для задачи достаточно. |
| Качество русского (опубликовано) | Ru Arena General: **59.2** (95% CI −2.1…+1.8) против Qwen3-1.7B 49.7 и no-reasoning Qwen3-1.7B 51.9 (карточка HF). MERA **36.8**, RuBIN **33.0** (таблица 4 в arXiv 2512.21580, строка «QVikhr(Qwen3)»). Отдельных замеров пунктуации/исправления ASR-ошибок нет — **не указано**. |
| Лицензия | **Apache-2.0** (тег HF `license: apache-2.0`, карточка). Коммерческое использование в приложении **разрешено**; требуется сохранить текст лицензии/NOTICE и указать изменения. Базовая Qwen3-1.7B — тоже Apache-2.0. Лицензия датасета `Vikhrmodels/GrandMaster2` отдельно не проверялась (для Apache-2.0 весов это не блокер, но зафиксировать в аудите). |
| Instruct/chat-шаблон | Да, Qwen3 ChatML: `<|im_start|>role\n...<|im_end|>`, генерация начинается с `<|im_start|>assistant`. В шаблоне есть блок ` thinking…</think>`; вариант «noreasoning» в примере карточки выдаёт пустой ` thinking\n\n</think>`. Для постпроцессора: использовать `/no_think` или вырезать блок. |
| Готовые форматы | **safetensors (BF16)** — основной; **GGUF** — официальный и community; **MLX** 4/8-bit. **ONNX — нет** (ни официального, ни в карточке). |
| ONNX-экспорт | Официального нет. Путь: `onnxruntime-genai` model builder (`python -m onnxruntime_genai.models.builder`) или `optimum-intel`/`optimum` → ONNX, затем int8/int4. Архитектура Qwen3 — decoder-only, GQA, RoPE, RMSNorm, SwiGLU, **QK-RMSNorm** (q_norm/k_norm). QK-norm и поддержку Qwen3 в текущей версии ORT GenAI нужно проверить перед планированием; при отсутствии — экспорт вручную/через `torch.onnx` (PyTorch только на билд-машине, не в приложении). |
| ONNX Runtime GenAI (Android) | Архитектурно совместим (Qwen2-подобный decoder + KV-cache), но **официальный генератор для Qwen3 не подтверждён** → вероятность средняя. ORT GenAI для Android существует (Java/Kotlin API, AAR). |
| MediaPipe LLM Inference (LiteRT) | **Не поддерживается официально**: список архитектур MediaPipe — Gemma/Gemma 2, Phi-2, Falcon, StableLM. Qwen3 в списке нет → низкая вероятность без конвертации/патча. |
| Streaming / token-by-token | Да, архитектурно (KV-cache, авторегрессия). В ORT GenAI — `Generator` со streaming API; в MediaPipe — streaming есть, но модель не поддерживается. |

### 1.2. Вероятность запуска на Android arm64 в разрешённых рантаймах

**Средняя.**

Обоснование: веса в safetensors и Apache-2.0 снимают юридический блокер; Qwen3 — стандартный
decoder, который в принципе экспортируется в ONNX. Но официального ONNX-артефакта нет,
а Qwen3-специфичные QK-RMSNorm/GQA требуют проверки в ORT GenAI model builder; MediaPipe
Qwen3 не поддерживает. Успех зависит от одного инженерного гейта — экспорт + int4/int8.

### 1.3. Подводные камни

- **Нет официального ONNX** → экспорт и квантизация — отдельная задача optimizer с проверкой
  численного совпадения логитов и качества на русской пунктуации.
- **Qwen3 QK-RMSNorm** может не покрываться готовыми fusion/квантизацией ORT GenAI; риск
  падения точности int4 на морфологии/пунктуации.
- **` thinking`-блок** в шаблоне: для постпроцессора обязательно отключать/вырезать, иначе
  модель будет тратить токены и ломать формат ответа.
- **GGUF/MLX — вне разрешённых рантаймов**; использовать только как источник весов для
  конвертации, не как runtime (иначе эскалация по HARD CONSTRAINTS).
- **Размер int8 ~1.8 ГБ** — ощутимо для среднего телефона вместе с ASR (~67 МБ) и
  диаризацией; int4 ~1 ГБ предпочтительнее, но требует замера качества.
- **Лицензия датасета GrandMaster2** не проверена; для Apache-2.0 весов это не меняет
  лицензию модели, но зафиксировать в лицензионном аудите.
- **Бенчмарки Ru Arena/MERA** измеряют знания/рассуждения, а не пунктуацию и исправление
  ASR-ошибок; собственного замера на целевой задаче нет.

---

## 2. Кандидат 2 — Gamayun 1.5B

### 2.1. Карточка

| Поле | Значение |
|---|---|
| Точное имя | **Gamayun 1.5B** (публичного HF-репозитория с весами не найдено) |
| URL | Статья: https://arxiv.org/abs/2512.21580 (v1 25.12.2025, v2 29.12.2025); HF paper page: https://huggingface.co/papers/2512.21580. **Ссылки на веса/код в статье и HF paper page нет** (`linkedModels: []`). |
| Владелец | Коллектив авторов: Alexander Podolskiy, Semen Molokov, Timofey Gerasin, Maksim Titov, Alexey Rukhovich, Artem Khrapov, Kirill Morozov, Evgeny Tetin, Constantine Korikov, Pavel Efimov, Polina Lazukova, Yuliya Skripkar, Nikita Okhotnikov, Irina Piontkovskaya, Meng Xiaojun, Zou Xueyi, Zhang Zhenhe. Аффилиация в статье не указана; публичного «владельца» модели/репозитория нет. |
| Базовая модель | **Обучена с нуля** (from scratch), не fine-tune. LLaMA-like архитектура. |
| Версия / дата | Статья: 25.12.2025 (v1), 29.12.2025 (v2). Версии весов нет — веса не опубликованы. |
| Размер весов | 1.47B параметров (1.5B + 0.2B tied input/output embeddings; в таблице статьи — 1.47B). BF16 ≈ **2.9 ГБ**; int8 ≈ **1.5 ГБ**; int4 ≈ **0.8–0.9 ГБ** (оценка). |
| Квантизации, которые реально есть | **Нет опубликованных** (ни BF16 safetensors, ни GGUF, ни ONNX). |
| Контекстное окно | **16 384** токена (расширено с 2 048 через ABF; RoPE base 10 000 → 500 000; +40B токенов annealing). |
| Качество русского (опубликовано) | MERA **38.1** (лучший среди 1–2B в статье), RuBIN **35.2**, среднее по русскому **50.7**; Belebele-ru 70.4, MMLU-ru 41.2, INCLUDE-ru 46.4, Arc_ru 34.9, Hellaswag_ru 42.4, XNLI-ru 47.8, Xstorycloze 68.3, Xwinograd 66.7 (arXiv 2512.21580, табл. 3–4). Замеров пунктуации/исправления ASR-ошибок нет — **не указано**. |
| Лицензия | **Не указано.** Лицензии на веса нет, т.к. веса не опубликованы. Лицензия arXiv (`non-exclusive-distrib/1.0`) относится к тексту статьи, **не к модели**. Коммерческое использование **не разрешено де-факто** — нет объекта лицензирования. |
| Instruct/chat-шаблон | Модель post-trained (SFT + DPO), но **шаблон не опубликован**. Вероятно LLaMA-3-style (использован LLaMA 3 tokenizer), но это предположение — **не указано**. |
| Готовые форматы | **Нет.** safetensors/GGUF/ONNX не опубликованы. |
| ONNX-экспорт | Теоретически возможен: LLaMA-like (SwiGLU, RoPE, RMSNorm pre-norm, 24 слоя, 16 голов, hidden 2048, LLaMA 3 tokenizer) → ORT GenAI model builder (Llama arch). Но **экспортировать нечего**, пока нет весов. |
| ONNX Runtime GenAI (Android) | Архитектура LLaMA-совместима → при наличии весов вероятность высокая. Сейчас — **нет артефактов**. |
| MediaPipe LLM Inference (LiteRT) | LLaMA-like в официальном списке MediaPipe нет (Gemma/Gemma 2, Phi-2, Falcon, StableLM) → низкая. |
| Streaming / token-by-token | Архитектурно да (KV-cache), но проверить не на чем. |

### 2.2. Вероятность запуска на Android arm64 в разрешённых рантаймах

**Низкая (фактически нулевая на 2026-11-06).**

Обоснование: веса не опубликованы, лицензии на модель нет, форматов/шаблона нет. Даже при
публикации LLaMA-like архитектура хорошо экспортируется в ONNX, но сейчас кандидат
непроверяем и юридически непригоден.

### 2.3. Подводные камни

- **Веса не опубликованы** — главный блокер; HF paper page не содержит linked models,
  поиск по HF API (`search=gamayun`) даёт только RL-модели постороннего аккаунта.
- **Лицензия отсутствует** — нельзя включать в APK даже как опциональную загрузку.
- **Нет chat-шаблона** — формат промпта неизвестен; воспроизводимость нулевая.
- **Нет квантизаций/ONNX** — весь путь экспорта придётся строить с нуля.
- **16k контекста** достаточно для реплик, но меньше, чем у Qwen (32k) — для длинных
  документов может потребоваться чанкинг.
- **LLaMA 3 tokenizer** (128k vocab) увеличивает embedding-слой; при int4 это заметная
  доля размера.
- **Бенчмарки MERA/RuBIN** — знания/культура, не пунктуация; релевантность задаче не измерена.

---

## 3. Фолбэки

### 3.1. Фолбэк 1 — Qwen2.5-1.5B-Instruct

| Поле | Значение |
|---|---|
| Точное имя | `Qwen/Qwen2.5-1.5B-Instruct` |
| URL | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct |
| Владелец | Qwen (Alibaba Cloud) |
| Базовая модель | `Qwen/Qwen2.5-1.5B` |
| Версия / дата | Создан 17.09.2024; последнее изменение 25.09.2024; sha `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Размер весов | 1.54B параметров (BF16). BF16 ≈ **3.1 ГБ**; int8 ≈ **1.6 ГБ**; int4 ≈ **0.9–1.0 ГБ** (оценка). |
| Квантизации | Официально safetensors BF16; community GGUF/ONNX/AWQ. Официального int8/int4 ONNX в репозитории нет, но ORT GenAI model builder умеет int4/int8. |
| Контекстное окно | **32 768** токенов (131 072 с YaRN). |
| Качество русского | MERA **35.7**, RuBIN **32.0**, среднее по русскому **49.8** (arXiv 2512.21580, табл. 3–4). Пунктуация/ASR-исправление — **не указано**. |
| Лицензия | **Apache-2.0** (тег HF, `LICENSE` в репозитории). Коммерческое использование разрешено; сохранить лицензию/NOTICE. |
| Instruct/chat-шаблон | Да, ChatML: `<|im_start|>system/user/assistant … <|im_end|>`; eos `<|im_end|>`. |
| Готовые форматы | safetensors (BF16); community GGUF, ONNX, AWQ. |
| ONNX-экспорт | **Официально поддерживается** ONNX Runtime GenAI model builder для Qwen2/Qwen2.5; также `optimum` → ONNX. |
| ONNX Runtime GenAI (Android) | **Высокая**: Qwen2/Qwen2.5 — в поддерживаемых архитектурах ORT GenAI; Android AAR есть. |
| MediaPipe LLM Inference | Официально Qwen не поддерживается → низкая. |
| Streaming | Да, ORT GenAI `Generator` streaming. |
| Вероятность запуска | **Высокая** (ORT GenAI). Подводные камни: RU чуть слабее QVikhr; нужен свой замер пунктуации; int4 может терять редкие слова/имена. |

### 3.2. Фолбэк 2 — Phi-3.5-mini-instruct

| Поле | Значение |
|---|---|
| Точное имя | `microsoft/Phi-3.5-mini-instruct` |
| URL | https://huggingface.co/microsoft/Phi-3.5-mini-instruct |
| Владелец | Microsoft |
| Базовая модель | Phi-3.5-mini (fine-tune) |
| Версия / дата | Создан 16.08.2024; последнее изменение 10.12.2025; sha `2fe192450127e6a83f7441aef6e3ca586c338b77` |
| Размер весов | 3.82B параметров (BF16). BF16 ≈ **7.6 ГБ**; int8 ≈ **3.8 ГБ**; int4 ≈ **2.1–2.3 ГБ** (оценка). |
| Квантизации | Официально safetensors BF16; community GGUF/ONNX. int4/int8 — через ORT GenAI/optimum. |
| Контекстное окно | **128 000** токенов. |
| Качество русского | В карточке — «multilingual» (20+ языков), но **RU-бенчмарки не опубликованы** → **не указано**. Ожидаемо слабее Qwen/QVikhr на русском; нужен замер. |
| Лицензия | **MIT** (тег HF, `LICENSE`). Коммерческое использование разрешено; сохранить copyright notice. |
| Instruct/chat-шаблон | Да: `<|system|>\n…<|end|>\n<|user|>\n…<|end|>\n<|assistant|>\n`; eos `<|endoftext|>`. |
| Готовые форматы | safetensors (BF16); community GGUF/ONNX. |
| ONNX-экспорт | ONNX Runtime GenAI поддерживает Phi-3/Phi-3.5 (model builder); `optimum` → ONNX. |
| ONNX Runtime GenAI (Android) | **Средняя–высокая**: Phi-3 поддержан; Phi-3.5 требует проверки версии ORT GenAI. |
| MediaPipe LLM Inference | Официально поддержан **Phi-2**, не Phi-3.5 → низкая/средняя. |
| Streaming | Да (ORT GenAI). |
| Вероятность запуска | **Средняя**: 3.8B → int4 ~2.1 ГБ, тяжело для среднего телефона; RU-качество не подтверждено. Подводные камни: размер, неизвестное RU-качество, поддержка Phi-3.5 в конкретной версии ORT GenAI. |

### 3.3. Дополнительно (не фолбэк, для сведения)

- `google/gemma-2-2b-it` — **нативно поддерживается MediaPipe LLM Inference** (LiteRT),
  это самый простой путь по рантайму. Но лицензия — **Gemma Terms of Use**, не OSI:
  коммерческое использование разрешено с ограничениями (prohibited use policy,
  распространение условий). Требует юридической проверки → не включён в основные фолбэки.
- `Vikhrmodels/QVikhr-2.5-1.5B-Instruct-r` (Qwen2.5-1.5B-based, GGUF есть) — потенциально
  лучший компромисс «RU-тюнинг + Qwen2-архитектура», но карточка в этом ревью не открывалась;
  проверить лицензию/дату перед включением.

---

## 4. Сравнительная таблица

| Кандидат | Размер int4 / int8 | Лицензия | Рантайм (разрешённый?) | RU-качество (опубликовано) | Готовность экспорта | Риск |
|---|---|---|---|---|---|---|
| **QVikhr-3-1.7B-Instruction-noreasoning** | ≈1.0 ГБ / ≈1.8 ГБ (оценка) | **Apache-2.0** — да | ONNX Runtime GenAI — средняя; MediaPipe — нет | Ru Arena 59.2; MERA 36.8; RuBIN 33.0 | safetensors + GGUF/MLX; **официального ONNX нет** → свой экспорт | **Средний**: Qwen3 QK-norm, нет ONNX, `think`-блок |
| **Gamayun 1.5B** | ≈0.8 ГБ / ≈1.5 ГБ (оценка) | **не указано** (веса не опубликованы) — **нельзя** | ONNX GenAI — теоретически; MediaPipe — нет | MERA 38.1; RuBIN 35.2; RU avg 50.7 | **нет артефактов** | **Критический**: нет весов, лицензии, шаблона |
| **Qwen2.5-1.5B-Instruct** (фолбэк) | ≈0.9 ГБ / ≈1.6 ГБ (оценка) | **Apache-2.0** — да | ONNX Runtime GenAI — **высокая**; MediaPipe — нет | MERA 35.7; RuBIN 32.0; RU avg 49.8 | safetensors; ORT GenAI model builder поддерживает Qwen2.5 | **Низкий–средний**: RU чуть слабее, нужен замер |
| **Phi-3.5-mini-instruct** (фолбэк) | ≈2.1 ГБ / ≈3.8 ГБ (оценка) | **MIT** — да | ONNX Runtime GenAI — средняя; MediaPipe — Phi-2 only | **не указано** (RU-бенчмарков нет) | safetensors; ORT GenAI поддерживает Phi-3/3.5 | **Средний–высокий**: размер 3.8B, RU не подтверждён |

---

## 5. Рекомендация

**Основной кандидат: `Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning` (Apache-2.0).**
Лучшее опубликованное RU-качество среди доступных 1–2B моделей (Ru Arena 59.2, MERA 36.8),
приемлемый размер (int4 ≈1 ГБ), есть instruct-шаблон и safetensors. Критический гейт —
собственный ONNX-экспорт (ORT GenAI model builder) + int4/int8 и замер на задаче
«пунктуация + исправление ASR-ошибок»; без этого гейта модель дальше не идёт.

**Фолбэк: `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0).**
Если экспорт Qwen3 не пройдёт или int4 деградирует: Qwen2.5 официально поддержан в ORT GenAI,
экспорт предсказуем, RU-качество близко (MERA 35.7 vs 36.8), размер тот же. Это страховка
по инженерному риску, а не по качеству.

**Gamayun 1.5B — отклонить** до публикации весов и лицензии; держать в watchlist
(сильные MERA/RuBIN, но непроверяем).

---

## 6. Лицензионные риски

| Модель | Лицензия | Статус для приложения |
|---|---|---|
| QVikhr-3-1.7B-Instruction-noreasoning | Apache-2.0 | **Разрешено**. Сохранить LICENSE/NOTICE, указать изменения. Проверить лицензию датасета GrandMaster2 (не блокирует Apache-2.0 весов, но зафиксировать). |
| Gamayun 1.5B | не указано (веса не опубликованы) | **Запрещено** до публикации лицензии. Лицензия arXiv — только на текст статьи. |
| Qwen2.5-1.5B-Instruct | Apache-2.0 | **Разрешено**. LICENSE/NOTICE. |
| Phi-3.5-mini-instruct | MIT | **Разрешено**. Copyright notice. |
| Gemma-2-2B-it (доп.) | Gemma Terms of Use (не OSI) | **Требует юридической проверки**: коммерческое использование разрешено, но с prohibited-use ограничениями и обязанностью распространять условия. |
| GGUF/MLX-конвертации QVikhr | Apache-2.0 (производные) | Формально лицензия та же, но **рантайм GGUF/MLX вне HARD CONSTRAINTS** → использовать только как источник весов для ONNX-конвертации. |

Общие правила:
- Для каждой модели фиксировать точный URL, commit sha, дату скачивания и хэш файла.
- Apache-2.0/MIT требуют сохранения уведомлений; для APK — отдельный экран «Лицензии».
- Любую модель без явной лицензии (Gamayun) нельзя включать даже как опциональную загрузку.

---

## 7. Источники (проверено 2026-11-06)

1. QVikhr-3-1.7B-Instruction-noreasoning — карточка HF: https://huggingface.co/Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning (обновление 30.05.2025; Apache-2.0; Ru Arena 59.2; база Qwen3-1.7B).
2. QVikhr GGUF — https://huggingface.co/Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning-GGUF ; MLX 4bit/8bit — соответствующие репозитории Vikhrmodels.
3. Qwen3-1.7B — https://huggingface.co/Qwen/Qwen3-1.7B (базовая модель; Apache-2.0; контекст 32k/131k YaRN).
4. Gamayun — arXiv:2512.21580, v1 25.12.2025, v2 29.12.2025: https://arxiv.org/abs/2512.21580 ; HTML: https://arxiv.org/html/2512.21580v2 ; HF paper page: https://huggingface.co/papers/2512.21580 (linkedModels пусто).
5. Qwen2.5-1.5B-Instruct — https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct (Apache-2.0; 1.54B; sha 989aa798…; ChatML).
6. Phi-3.5-mini-instruct — https://huggingface.co/microsoft/Phi-3.5-mini-instruct (MIT; 3.82B; sha 2fe19245…; 128k).
7. ONNX Runtime GenAI (model builder, поддерживаемые архитектуры, Android): https://github.com/microsoft/onnxruntime-genai ; https://onnxruntime.ai/docs/genai/
8. MediaPipe LLM Inference (поддерживаемые модели: Gemma/Gemma 2, Phi-2, Falcon, StableLM): https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference

---

## 8. Открытые вопросы / что проверить перед следующим шагом

1. **Экспорт QVikhr в ONNX**: поддерживает ли текущая версия ORT GenAI model builder Qwen3
   (QK-RMSNorm, GQA)? Если нет — оценить ручной экспорт и стоимость поддержки.
2. **Замер на целевой задаче**: пунктуация + исправление ASR-ошибок + флаг «сомнительное
   словосочетание» на русском наборе; сравнить QVikhr int4/int8 vs Qwen2.5 int4/int8.
3. **Память/скорость на реальном телефоне**: int4 ~1 ГБ + KV-cache + ASR/диаризация —
   уложиться в бюджет Т3/Т4 (значения ещё не заданы).
4. **Gamayun**: мониторить публикацию весов/лицензии; без этого — не рассматривать.
5. **Лицензия датасета GrandMaster2** — зафиксировать в лицензионном аудите.
