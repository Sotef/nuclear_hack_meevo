# Восстановление пунктуации русского текста (text → punctuated text) для Android arm64

Дата: 2026-11-07. Метод: только карточки HF/API (веса не скачивались). Бюджет: 12 webfetch.

## Критерии
- вход: голый русский текст (lowercase, без пунктуации), без аудио;
- выход: `. , ? ! : ; — - ... ?!` + регистр (Capitalized/UPPER);
- размер < 500 МБ, экспорт в ONNX (token-classification → logits `[B, T, C]`);
- лицензия, разрешающая использование в приложении.

## Таблица кандидатов

| # | Модель / URL | Архитектура | Размер | Лицензия | Языки | Вход → выход | ONNX | Вердикт |
|---|---|---|---|---|---|---|---|---|
| 1 | [RUPunct/RUPunct_small](https://huggingface.co/RUPunct/RUPunct_small) | BERT token-classification, 33 метки (3 регистра × 11 знаков) | 116 МБ fp32 (~29M params) | **MIT** | ru | текст → метка на слово; декод `process_token` | да, есть готовый INT8: [ekhodzitsky/rupunct-small-onnx](https://huggingface.co/ekhodzitsky/rupunct-small-onnx) — **29 МБ**, вход `input_ids/attention_mask/token_type_ids`, выход `logits [B,T,33]` | **ГОДИТСЯ** |
| 2 | [RUPunct/RUPunct_medium](https://huggingface.co/RUPunct/RUPunct_medium) | ELECTRA token-classification | 340 МБ | MIT | ru | то же | экспорт стандартный (torch.onnx), готового нет | требует проверки (качество выше, но 340 МБ) |
| 3 | [RUPunct/RUPunct_big](https://huggingface.co/RUPunct/RUPunct_big) | BERT-large token-classification | 711 МБ | MIT | ru | то же | экспорт возможен | **НЕТ** (>500 МБ) |
| 4 | [Horizon-Labs/multilingual-punctuation-truecase-small](https://huggingface.co/Horizon-Labs/multilingual-punctuation-truecase-small) | mmBERT-small, 18 joint labels (знак+регистр) | 140M params; ONNX int8 **268 МБ** | **Apache-2.0** | 88, вкл. ru | lowercase текст → пунктуация + Cap/UPPER за один проход | **ONNX уже в репо** (`onnx/model_quantized.onnx`), transformers.js | **ГОДИТСЯ (запасной)** |
| 5 | [Horizon-Labs/punctuation-restoration-small](https://huggingface.co/Horizon-Labs/punctuation-restoration-small) | mmBERT-small, token-classification | 140M params | Apache-2.0 | 88, вкл. ru | текст → только пунктуация (регистр не меняет) | ONNX в репо | годится, но без регистра |
| 6 | [valkayuh/dewpoint](https://huggingface.co/valkayuh/dewpoint) | XLM-RoBERTa-large ансамбль, 79 языков | ~560M+ | MIT | 79, вкл. ru | текст → пунктуация + truecase | ONNX есть | **НЕТ** (тяжёлая, >500 МБ) |
| 7 | [denis-berezutskiy-lad/lad_transcription_bert_ru_punctuator](https://huggingface.co/denis-berezutskiy-lad/lad_transcription_bert_ru_punctuator) | NeMo BERT token-classification | не указано | Apache-2.0 | ru | текст → пунктуация | нет, нужен NeMo-рантайм | требует проверки (сложный экспорт) |
| 8 | dchaplinsky/punctuator | — | — | — | — | — | — | **НЕДОСТУПНА**: HF API отдаёт 401 (удалена/приватна) |

## Метрики (только заявленные на карточках)
- **RUPunct_small**: F1 на карточке **не указан**; есть demo Space `RUPunct/RUPunct-demo`. Заявлено: «самая маленькая, для CPU/скорости».
- **Horizon-Labs truecase-small**: ru — punctuation F1 **0.913**, casing F1 **0.928**, casing accuracy 0.982 (FLORES, из таблицы карточки). Это выше текущего LLM-варианта (0.757).
- **Horizon-Labs punctuation-restoration-small**: ru F1 ≈0.91 (та же серия/данные; отдельная per-language строка на карточке truecase для punct-only не приведена — цифра ориентировочная).
- **dewpoint**: заявлен как лучший из протестированных, но размер не для телефона.

## Как экспортировать / запустить
1. **RUPunct_small (рекомендуемый путь)**: готовый `rupunct_small_int8.onnx` (29 МБ) + `tokenizer.json` (WordPiece). Токенизатор обязательно `strip_accents=False, add_prefix_space=True`. Инференс: `logits [B,T,33]` → для каждого слова берём метку первого суб-токена (`aggregation="first"`) → декод из 33 классов (11 знаков × LOWER/UPPER/UPPER_TOTAL). ONNX Runtime Mobile, int64 входы. Никакого Python в рантайме (в gigastt используется Rust `tokenizers`).
2. **Horizon-Labs**: `onnx/model_quantized.onnx` уже в репо, стандартный token-classification; метка слова = метка последнего токена; вход предварительно lowercased; декод `truecase.py` (18 labels `<mark>|<case>`).
3. Своя конвертация (если нужен fp16/иной opset): `torch.onnx.export` из `AutoModelForTokenClassification`, opset 17, dynamic axes `batch/sequence`.

## Оценка скорости на телефоне (оценка, не замер)
- RUPunct_small INT8, seq≤128, 1 поток ARM: ~15–40 мс/сегмент (29M params, BERT-small). Против ~2500 мс у LLM — ускорение ~60–150×.
- Horizon-Labs small INT8 (140M): ~60–150 мс/сегмент, память ~270 МБ — тяжелее, но всё ещё приемлемо.
- Реальный замер обязателен: качество RUPunct на «сыром» ASR-тексте без регистра не подтверждено метрикой.

## Риски
- RUPunct обучен, вероятно, на тексте с регистром; на полностью lowercase входе возможны ошибки капитализации. Проверить на своём корпусе.
- WordPiece + `add_prefix_space` — частая причина склейки слов (та же болезнь, что у CT-Transformer). Тестировать на кириллице с пунктуацией и без.
- 33-классовый декод чувствителен к агрегации суб-токенов (`first` vs `last`).

## Рекомендация
1. **Для замера: `ekhodzitsky/rupunct-small-onnx`** (INT8, 29 МБ, MIT, ru-only, пунктуация + регистр, готовый ONNX). Замерить punct F1 на своём валидационном наборе и латентность на arm64.
2. **Запасной: `Horizon-Labs/multilingual-punctuation-truecase-small`** (Apache-2.0, 268 МБ int8, ru punct F1 0.913 / casing F1 0.928 по карточке, ONNX из коробки) — если RUPunct_small провалит качество.
3. Если оба < текущего LLM (0.757) — остаёмся на QVikhr int4.

## Итог
Ни один кандидат не имеет опубликованного F1 именно для RUPunct, поэтому «гарантированной замены» нет. Но найден лёгкий (29 МБ, MIT) готовый ONNX-вариант русского пунктуатора — это лучший кандидат на замер; запасной — Apache-2.0 мультиязычная модель с подтверждённым ru F1 0.913.
