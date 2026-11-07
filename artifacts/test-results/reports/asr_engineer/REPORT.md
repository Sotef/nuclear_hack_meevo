# ASR baseline: два кандидата sherpa-onnx на русском

Дата: 2026-11-06. Автор: asr-engineer.
Набор: `data/test` — 13 WAV (моно, 16 кГц, PCM16), 310.641 с, 497 слов после RU-нормализации.
Метрика WER: только `tools/eval_asr` 1.0.0 (micro и macro, per-record).
Окружение: Windows, CPU (десктоп), Python 3.12.10, `.venv`.

## 1. Итоговая сравнительная таблица

| Параметр | A: `sherpa-onnx-zipformer-ru-2024-09-18` | B: `sherpa-onnx-small-zipformer-ru-2024-09-18` |
|---|---|---|
| WER micro (eval_asr) | **0.175050** (87 ошибок / 497) | **0.146881** (73 / 497) |
| WER macro (eval_asr) | 0.169596 | 0.144337 |
| S / D / I | 43 / 34 / 10 | 49 / 14 / 10 |
| RTF (среднее 2 прогонов, CPU, 1 поток) | **0.0235** (7.300 с / 310.641 с) | **0.0189** (5.886 с / 310.641 с) |
| Время инференса, прогон 1 / 2 | 7.386 с / 7.215 с | 6.057 с / 5.716 с |
| Загрузка модели | 2.546 с | 1.528 с |
| Пиковая RAM процесса (psutil RSS) | 399.95 МБ | 350.81 МБ |
| Размер int8-комплекта | 66.8 MiB (70 007 274 Б) | 25.7 MiB (26 911 058 Б) |
| Тайминги | token-level + word-level + сегменты (эвристика) | token-level + word-level + сегменты (эвристика) |
| Streaming | нет (offline transducer) | нет (offline transducer) |
| Лицензия | Apache-2.0 | Apache-2.0 |
| Версия модели | 2024-09-18 | 2024-09-18 |
| Библиотека | sherpa-onnx 2.0.1 (pip, CPU) | sherpa-onnx 2.0.1 (pip, CPU) |

Вывод по цифрам: на этом наборе **B (small) лучше и по WER, и по RTF, и по размеру/RAM**.
A (big) проигрывает на аудиокнижном чистом тексте, но по опубликованным данным Vosk
большая модель заметно лучше на спонтанной/телефонной речи — этот домен в наборе не представлен.

## 2. Метод (одинаковый для A и B)

- Инструмент: `tools/run_asr/run_asr.py` 1.0.0, backend `sherpa-onnx`, `--precision int8`,
  `--threads 1`, `--repeat 2`, `--max-chunk-sec 20`, greedy search, `dither=0`.
- Декодер: `decoder.onnx` (fp32) — как в официальной команде sherpa-onnx; encoder/joiner — int8.
- WER: `tools/eval_asr/eval_asr.py` 1.0.0, `--hyp tools/eval_asr/hyp/<candidate>`.
- RTF = среднее wall-time инференса двух полных прогонов / суммарная длительность аудио.
  Загрузка модели в RTF не входит (учтена отдельно).
- Пик RAM: `psutil.Process().memory_info().rss`, максимум после каждой записи; включает
  Python + onnxruntime + модель, поэтому это не бюджет Android.
- Детерминированность: greedy, `dither=0`, seed не используется. Проверено: повторный
  прогон A в отдельный каталог дал **0 отличий** в 13/13 `.txt`.

### Команды запуска

```powershell
# A
.venv\Scripts\python.exe tools\run_asr\run_asr.py `
  --model models\sherpa-onnx-zipformer-ru-2024-09-18 `
  --manifest data\test\manifest.jsonl --out tools\eval_asr\hyp\zipformer_ru `
  --backend sherpa-onnx --precision int8 --threads 1 --repeat 2 --max-chunk-sec 20 --force
.venv\Scripts\python.exe tools\eval_asr\eval_asr.py `
  --manifest data\test\manifest.jsonl --hyp tools\eval_asr\hyp\zipformer_ru `
  --report reports\asr_engineer\eval_zipformer_ru

# B
.venv\Scripts\python.exe tools\run_asr\run_asr.py `
  --model models\sherpa-onnx-small-zipformer-ru-2024-09-18 `
  --manifest data\test\manifest.jsonl --out tools\eval_asr\hyp\small_zipformer_ru `
  --backend sherpa-onnx --precision int8 --threads 1 --repeat 2 --max-chunk-sec 20 --force
.venv\Scripts\python.exe tools\eval_asr\eval_asr.py `
  --manifest data\test\manifest.jsonl --hyp tools\eval_asr\hyp\small_zipformer_ru `
  --report reports\asr_engineer\eval_small_zipformer_ru
```

## 3. WER по записям

| id | длит., с | ref слов | A WER | B WER |
|---|---:|---:|---:|---:|
| ruls_0001 | 19.88 | 30 | 0.1000 | 0.1667 |
| ruls_0002 | 19.79 | 34 | 0.4412 | 0.1765 |
| ruls_0003 | 19.76 | 32 | 0.1250 | 0.1250 |
| ruls_0004 | 19.58 | 35 | 0.0857 | 0.0571 |
| ruls_0005 | 19.52 | 44 | 0.1818 | 0.2045 |
| ruls_0006 | 19.46 | 34 | 0.0000 | 0.0000 |
| ruls_0007 | 19.39 | 42 | 0.0952 | 0.0714 |
| ruls_0008 | 19.14 | 28 | 0.0357 | 0.0714 |
| ruls_0009 | 19.06 | 30 | 0.2667 | 0.2000 |
| ruls_0010 | 18.98 | 36 | 0.1111 | 0.2222 |
| diar_0001 | 46.13 | 58 | 0.1552 | 0.1034 |
| diar_0002 | 33.51 | 52 | 0.2500 | 0.1923 |
| diar_0003 | 36.44 | 42 | 0.3571 | 0.2857 |
| **micro** | 310.64 | 497 | **0.175050** | **0.146881** |
| **macro** | | | 0.169596 | 0.144337 |

Полные отчёты: `reports/asr_engineer/eval_zipformer_ru.{json,txt}`,
`reports/asr_engineer/eval_small_zipformer_ru.{json,txt}`.

## 4. Тайминги

Обе модели отдают token-level timestamps (`result.timestamps`). `run_asr` сохраняет
`tools/eval_asr/hyp/<candidate>/timestamps.json`:

- `tokens[]` — `{token, start}` (сырые token-level метки sherpa-onnx);
- `words[]` — `{text, start, end}` (склейка BPE-токенов в слова; end = время последнего токена);
- `segments[]` — `{start, end, text}` (эвристика: разрез при паузе ≥ 0.5 с между словами).

Важно: `end` слова — это время начала последнего токена, а не конца звучания; для
OUTPUT CONTRACT сегменты нужно будет уточнять (merge-engineer). Сегментация — эвристика,
не разметка модели.

## 5. Ограничения (явные)

1. **Лимит длины входа ~20 с.** Обе модели падают с ошибкой Reshape на аудио длиннее ~20 с
   (проверено: 20 с — OK, 21 с — FAIL). Поэтому `run_asr` режет записи на куски по 20 с
   без перекрытия. Это может терять/дублировать слова на границах; для diar_* (33–46 с)
   границ 1–2 на запись. Обе модели резались одинаково, сравнение честное, но абсолютный
   WER завышен относительно инференса целой записи.
2. **Домен набора.** 10 записей — чтение поэтических аудиокниг (Пушкин), один диктор,
   чистая студийная речь. 3 записи — синтетический TTS (2/3/4 «спикера»), не живые люди.
   Нет спонтанной речи, телефонии, шума, дальнего микрофона. WER на этом наборе **не
   переносится** на реальные разговоры.
3. **Скорость и RAM — десктопный CPU, не Android.** RTF 0.019–0.024 и пик 350–400 МБ
   нельзя использовать как бюджет телефона; для устройства нужен profiler.
4. **Пик RAM** включает Python + onnxruntime + веса; на Android цифра будет другой.
5. **Streaming отсутствует** у обеих моделей (offline transducer). «Псевдо-streaming»
   возможен только через VAD-нарезку + offline-декодирование.
6. **Лицензия** проверена по карточке Vosk models (Apache-2.0) и документации sherpa-onnx;
   отдельного юридического аудита весов не проводилось.

## 6. Рекомендация baseline

**Baseline-кандидат: B — `sherpa-onnx-small-zipformer-ru-2024-09-18` (int8, 25.7 MiB).**
Причины: на одном наборе и одном методе даёт лучший WER (0.1469 vs 0.1751), меньший RTF
(0.0189 vs 0.0235), меньший размер (25.7 vs 66.8 MiB) и меньший пик RAM (351 vs 400 МБ).

**A оставить как кандидата на следующий раунд** для спонтанной/телефонной речи: по
опубликованным WER Vosk большая модель существенно лучше small на calls/youtube, а этот
домен в текущем наборе отсутствует. Решение A vs B нельзя считать окончательным до
теста на реальных разговорах.

## 7. Что «не измерено»

- WER на спонтанной речи, телефонии, шуме — **не измерено** (нет данных в наборе).
- RTF и пик RAM на реальном Android-устройстве — **не измерено** (нужен profiler).
- WER без 20-секундной нарезки — **не измерено** (модели не принимают длинный вход).
- Влияние `decoder.int8` vs `decoder.onnx` на WER — **не измерено** (использован fp32-декодер
  по официальной рекомендации).
- Качество word-level таймингов (точность границ слов) — **не измерено** (нет референса
  с временными метками).
