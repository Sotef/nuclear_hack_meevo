# Журнал решений — data-eval

## 2026-11-06

### Контекст
Проект стартует с нуля: в `D:\Downloads_D\nuclear_hack` был только `rules.md`.
Задача: тестовый набор русской речи с эталонными транскрипциями, инструменты
prep_audio / eval_asr / eval_diar, журнал и отчёт. Python 3.12.10,
интерпретатор проекта — `.venv\Scripts\python.exe`.

### Решения по данным

1. **RuLS (istupakov/russian_librispeech) — основной реальный набор.**
   - Причина: HF dataset card `license: cc-by-4.0`, аудио 16 кГц, есть поле `text`,
     parquet с встроенным аудио; test-шард 298 МБ, 1352 примера.
   - Взято 10 самых длинных уникальных клипов (18.98–19.88 с), суммарно 194.56 с.
   - Ограничение: это чтение поэтических аудиокниг LibriVox (Пушкин), не разговорная речь.

2. **ivkond/synthetic-speech-diarization-ru — многоговорящие записи.**
   - Причина: HF dataset card `license: mit`, 16 кГц, 2000 треков, есть `num_speakers`,
     `speakers` (start/end/speaker_id/text), `has_overlaps`; parquet 76.8 МБ.
   - Взято 3 уникальных трека: 2, 3 и 4 спикера (46.132 / 33.507 / 36.441 с),
     суммарно 116.08 с. Из `speakers` сгенерированы RTTM-референсы.
   - Ограничение: это TTS-синтез, а не живые люди; сегменты в `speakers` последовательные,
     реальных пересечений в RTTM нет (флаг `has_overlaps` относится к `simultaneous_segments`).

3. **Golos — отклонён.**
   - В карточке `bond005/sberdevices_golos_10h_crowd` лицензия `other` и ссылка на
     «Public license with attribution and conditions reserved», а не CC-BY-4.0.
     Трактовать как CC-BY без юридической проверки не стали.
   - Дополнительно: минимальный parquet-шард 98–400 МБ, избыточно для 3–10 минут.

4. **Mozilla Common Voice ru — отклонён.**
   - Официальный датасет gated; доступные HF-зеркала в webdataset: test-шард 924 МБ
     (`Sh1man/common_voice_21_ru`), train/validate ещё больше. Для 3–10 минут нецелесообразно.

5. **CSS10 / silero-vad / open_stt — не использованы.**
   - CSS10: GitHub-репозиторий Kyubyong/css10 не содержит основного аудио (в дереве
     только MOS-референсы и тексты), скачивать нечего без внешних зеркал.
   - silero-vad: примеры VAD без эталонных транскрипций.
   - open_stt: репозиторий моделей/инструментов, а не размеченный аудиокорпус.

6. **Реальные человеческие записи с 2+ говорящими и транскрипцией — не найдено.**
   - Искали HF datasets по запросам `diarization`, `russian diarization`, `golos`,
     `russian speech`, `podlodka`. Найдены либо коммерческие/gated наборы, либо
     синтетика. Podlodka speech (`bond005/podlodka_speech`) имеет parquet с аудио и
     транскрипцией, но в карточке нет лицензии — не взяли.
   - Поэтому многоговорящая часть набора — синтетическая (MIT), с честной пометкой в отчёте.

### Итоговый набор
- 13 записей, 310.64 с (5.18 мин), WAV PCM16 / моно / 16 кГц.
- 10 × RuLS (CC-BY-4.0, 1 спикер) + 3 × synthetic diarization (MIT, 2/3/4 спикера).
- `data/test/manifest.jsonl`: 13 строк, поля id, path, duration_sec, text_path,
  source, license, speakers_count (+ rttm_path/conversation_type/has_overlaps для diar).
- 497 слов после RU-нормализации.

### Решения по инструментам

- **prep_audio 1.0.0** — soundfile + numpy, без ffmpeg (в системе ffmpeg не найден).
  Ресемплинг линейной интерполяцией; для исходников 16 кГц не применялся.
  Защита от перезаписи без `--force` (код 2).
- **eval_asr 1.0.0** — чистый Python, word-level Levenshtein с backtrace (S/D/I),
  RU-нормализация: lower, ё→е, пунктуация → пробел, схлопывание пробелов.
  Поддержка `--hyp` (папка `<id>.txt` или один файл) и `--hyp-json`.
- **eval_diar 1.0.1** — frame-based DER, шаг 10 мс, collar 0.25 с по границам
  референса, speaker-time учёт перекрытий, brute-force mapping для ≤7 спикеров.
  Версия 1.0.1: исправлен учёт перекрытий (в 1.0.0 пропуск перекрывающегося
  спикера не попадал в ошибку; теперь overlap DER на тесте 0.5).

### Самопроверка (факты)

- prep_audio: 13/13 OK, 310.641 с; без `--force` → exit 2; с `--force` → exit 0.
- eval_asr: ref→WER 0 (497 слов); 1 замена→WER 0.002012 micro, 0.033333 для ruls_0001;
  `--hyp-json`→WER 0; несуществующий путь→exit 2.
- eval_diar: 3 RTTM ref vs ref→DER 0 (9040 кадров); удалён первый сегмент→DER 0.065819;
  handcrafted overlap ref vs ref→DER 0 (98 overlap-кадров); пропуск перекрывающегося
  спикера→overlap DER 0.5, overall 0.284058.

### Версии зависимостей
- Python 3.12.10
- soundfile 0.14.0
- numpy 2.5.4
- pyarrow 26.0.1 (только извлечение parquet)
- huggingface_hub 2.4.0 (загрузка)
- requests 2.34.2 (поиск/карточки)

### Открытые вопросы / что дальше
- Нужен реальный русский многоговорящий набор с RTTM/транскрипцией и открытой
  лицензией — иначе DER на реальных данных останется «не измерено».
- Для baseline ASR стоит добавить спонтанную/телефонную речь и шумовые условия.
- WER/DER моделей не измерялись: это зона asr-engineer / diar-engineer.

---

# Журнал решений — diar-engineer

## 2026-11-06

### Контекст
Baseline диаризации на `data/test`: 3 синтетические многоговорящие записи
(2/3/4 спикера, MIT) + 10 односпикеровых RuLS (CC-BY-4.0). Оценка — только
`tools/eval_diar` 1.0.1, collar 0.25. Python 3.12.10, `.venv`.

### Решения

1. **Рантайм — sherpa-onnx 2.0.1 (уже был в .venv).**
   - Взяты готовые ONNX-модели из GitHub releases `k2-fsa/sherpa-onnx`:
     pyannote segmentation-3.0 (fp32, MIT), 3D-Speaker CAM++ (Apache-2.0),
     TitaNet-small (CC-BY-4.0, только диагностика).
   - Архив segmentation распакован через Python `tarfile` (в системе нет bzip2).
   - Модели и хэши — `models/diar/MANIFEST.md`.

2. **Пайплайн:** pyannote segmentation-3.0 → эмбеддинги по сегментам →
   `FastClustering` → RTTM. Реализован как `tools/run_diar` 1.0.0
   (CLI: `--manifest --out --threshold --oracle-num-speakers --num-speakers`).
   - `OfflineSpeakerDiarization` не даёт менять clustering config после создания,
     поэтому диаризатор пересоздаётся на каждую запись (инициализация вне замера RTF).

3. **Выбор порога A.** Короткий sweep на 3 diar-записях:
   0.3 → DER 0.580 (5/5/9 спикеров), 0.5 → 0.541 (3/5/6),
   0.7 → **0.480** (2/3/5), 0.9 → 0.652 (1/2/3). Выбран 0.7.

4. **A vs B (одна переменная).** A — threshold 0.7; B — oracle `num_clusters`
   из `speakers_count` manifest. Всё остальное одинаково.
   - A: DER 0.480088, 0/10 ложных спикеров, RTF 0.0775.
   - B: DER 0.551991, 0/10, RTF 0.0770.
   - B хуже из-за `diar_0003`: при `num_clusters=4` FastClustering вернул 3
     спикера и DER 0.812 (у A — 5 спикеров, DER 0.573).

5. **Диагностика C (вне A/B).** TitaNet-small + threshold 0.7 на 3 diar:
   DER 0.503429 (хуже CAM++). На 10 односпикеровых не запускался.

6. **Дополнительная проверка D.** CAM++ threshold 0.7 + `min_duration_on/off=0.1`:
   DER 0.480088 (без изменений; miss даже +24 кадра). Значит пропуски — не эффект
   min-duration фильтра, а паузы внутри референсных реплик TTS.

### Проблемы / наблюдения
- `miss` = 3052/9040 (≈34%) одинаков у A/B/C: сегментатор не считает речью
  внутренние паузы TTS, а RTTM размечает целые реплики. Это главный вклад в DER.
- Overlap-DER не измерен: в ref RTTM 0 overlap-кадров (синтетика без пересечений).
- `diar_0003` — самая тяжёлая запись (4 спикера, 9 сегментов); oracle-режим
  не гарантирует заданное число кластеров.
- RTF 0.077 — десктопный Python/CPU, не Android.

### Версии
- Python 3.12.10, sherpa-onnx 2.0.1, numpy 2.5.4, soundfile 0.14.0
- eval_diar 1.0.1, run_diar 1.0.0

### Что дальше
- Проверить A на реальных русских многоговорящих записях с перекрытиями.
- Если нужен oracle-режим — разобраться, почему FastClustering возвращает
  меньше кластеров, чем `num_clusters`.
- Квантизация/Android-замер — зона optimizer/profiler.

---

## 2026-11-06 — asr-engineer: baseline ASR (2 кандидата sherpa-onnx)

### Решения по кандидатам
- Взяты оба кандидата из `reports/model_scout/CANDIDATES.md` (A: zipformer-ru, B: small-zipformer-ru),
  оба Apache-2.0, int8, offline transducer. Замена на Vosk/whisper.cpp не понадобилась.
- Скачаны архивы `asr-models` с GitHub k2-fsa/sherpa-onnx, распакованы в `models/`.
  Манифест с URL, лицензией, размерами и sha256 — `models/MANIFEST.md`.
- Установлены `sherpa-onnx 2.0.1` и `psutil 7.2.2` в `.venv`.
- Новый инструмент: `tools/run_asr/run_asr.py` 1.0.0 (прогон ASR по манифесту, hyp + timestamps + metrics).
  Запись добавлена в `tools/TOOLS.md`.

### Что не получилось / подводные камни
- Модели падают с ошибкой ONNX Reshape на аудио длиннее ~20 с (20 с — OK, 21 с — FAIL).
  Причина — фиксированная длина входа в экспортированном encoder. Решение: нарезка по 20 с
  в `run_asr` (`--max-chunk-sec 20`), без перекрытия. Это влияет на WER на границах.
- Первая версия `run_asr` использовала `decoder.int8.onnx`; официальная команда sherpa-onnx
  использует `decoder.onnx` (fp32). Переключено на fp32-декодер.
- Первая версия склейки BPE-токенов в слова вставляла пробел между всеми токенами
  («не ю» вместо «нею»). Исправлено: продолжение слова клеится без пробела.
- `tar -xjf` в Windows не сработал (нет bzip2); распаковка сделана через Python `tarfile` (bz2).

### Результаты (data/test, 13 записей, 310.641 с, eval_asr 1.0.0)
- A `sherpa-onnx-zipformer-ru-2024-09-18`: WER micro 0.175050, macro 0.169596;
  RTF 0.0235 (2 прогона: 7.386 / 7.215 с), пик RSS 399.95 МБ, загрузка 2.546 с.
- B `sherpa-onnx-small-zipformer-ru-2024-09-18`: WER micro 0.146881, macro 0.144337;
  RTF 0.0189 (6.057 / 5.716 с), пик RSS 350.81 МБ, загрузка 1.528 с.
- Детерминированность: greedy, dither=0; повторный прогон A — 0 отличий в 13/13 txt.
- Тайминги: token-level есть у обеих; сохранены token/word/segment в `timestamps.json`.

### Вывод
- Baseline-кандидат: **B (small)** — лучше WER, RTF, размер и RAM на этом наборе.
- A оставить для следующего раунда на спонтанной/телефонной речи (домен в наборе отсутствует).
- Не измерено: WER на реальных разговорах/шуме, RTF и RAM на Android, WER без нарезки 20 с.
- Отчёт: `reports/asr_engineer/REPORT.md`.

---

## 2026-11-06 — merge-engineer: слияние ASR + диаризация (baseline-0.1)

### Решение по алгоритму слияния
- Вход: `tools/eval_asr/hyp/small_zipformer_ru/timestamps.json` (сегменты ASR с таймингами)
  + `reports/diar_engineer/hyp/A_campplus_thr0.7/<id>.rttm` (конфиг A, CAM++ + pyannote-seg-3.0
  + FastClustering thr 0.7) + `data/test/manifest.jsonl` (duration_sec).
- Для каждого ASR-сегмента `[s, e]` и каждого RTTM-сегмента спикера `sp` считается
  `overlap(sp) = Σ max(0, min(e, re) − max(s, rs))`. Побеждает спикер с максимальной
  суммарной длительностью пересечения; при равенстве — лексикографически меньший id.
- Если пересечения нет ни с одним RTTM-сегментом (overlap = 0), берётся спикер ближайшего
  по времени RTTM-сегмента (расстояние = зазор между интервалами). Это fallback, он
  учитывается в статистике отдельно.
- `start`/`end` берутся из ASR-таймингов; нулевые по длине сегменты (`end == start`,
  5 шт. из-за того, что `words[].end` = время начала последнего токена) расширяются
  на 0.01 с, чтобы выполнить контракт `start < end`. `confidence = 0.0` — «не измерено»,
  модель уверенности не отдаёт. Сегменты сортируются по `start`.
- Результат: 13 JSON в `output/baseline/`, 84 сегмента; 2 сегмента без пересечения
  (fallback), 3 с малым пересечением (< 0.1 с), 5 расширенных нулевых. Валидация
  OUTPUT CONTRACT — 13/13; повторный прогон — 0 отличий SHA256.
- Инструмент: `tools/run_pipeline/merge.py` 0.1 (одна CLI-команда, `--help`, `--force`,
  `--validate` / `--validate-only`), запись добавлена в `tools/TOOLS.md`.

### Пометка оркестратора
bootstrap-цикл: лимит 2 тулз/цикл превышен из-за стартового набора; новые инструменты
заморожены до следующего цикла.

---

## 2026-11-06 — reviewer: независимая проверка baseline

- Вердикт: **PASS WITH NOTES**; замечаний: **6 некритичных, 0 критичных**.
- Перепроверено: WER small 0.146881 / big 0.175050; DER A 0.480088; FA 0/10;
  13/13 JSON valid; повторный merge — 0 отличий SHA256.
- Отчёт: `reports/reviewer/REVIEW.md`.

---

## 2026-11-06 — data-eval: eval-набор и харнесс LLM-постпроцессора

### Источник пунктуации и лицензия
- Рассматривался **UD ru-SynTagRus** (рекомендация из задания), но в
  `LICENSE.txt` репозитория UD_Russian-SynTagRus указана **CC BY-NC-SA 4.0**
  (NonCommercial). Для приложения это ограничение неприемлемо, поэтому
  источник отклонён.
- Выбран **Tatoeba Russian sentences** (`rus_sentences.tsv.bz2`,
  https://downloads.tatoeba.org/exports/per_language/rus/rus_sentences.tsv.bz2).
  Terms of use Tatoeba: для текстовых предложений по умолчанию действует
  **CC-BY 2.0 FR** (атрибуция; коммерческое использование разрешено).
  Экспорт не содержит per-sentence лицензию, поэтому записана консервативная
  `CC-BY-2.0-FR`; атрибуция Tatoeba contributors обязательна.
- Из 315 600 предложений, прошедших фильтры (кириллица, 5–30 слов, ≥2 знака
  пунктуации, финальный `.`/`!`/`?`, дедупликация по нижнему регистру),
  детерминированно (`seed=42`) отобрано **300** пар. 50 помечены `clean=true`
  для false-alarm proxy.
- Correction-набор: **13** пар из существующих `data/test/<id>.txt` и
  `tools/eval_asr/hyp/small_zipformer_ru/<id>.txt` (baseline WER 0.146881,
  497 слов). Новые данные не скачивались.

### Инструмент
- Создан **один** новый инструмент `tools/eval_postproc/eval_postproc.py`
  1.0.0 (лимит цикла соблюдён). Сторонних зависимостей нет, только stdlib.
- Метрики: punct micro/macro P/R/F1 по знакам `.,!?;:—–…`; boundary P/R/F1;
  WER before/after + delta + improved/worsened/unchanged; false-alarm rate и
  плотность doubtful-флагов на 50 чистых предложениях; покрытие известных
  ASR-ошибок по surface form.
- Self-test: hyp=ref → punct F1 **1.000000**, delta WER **0.000000**
  (correction hyp=ASR); correction hyp=ref → delta WER **-0.146881**
  (improved 12/13); удаление запятых → F1 **0.675000**; флаги →
  false-alarm **0.100000**, error coverage **0.092437**. Отчёты в
  `reports/postproc_eval/`.
- Запись в `tools/TOOLS.md` добавлена; спецификация —
  `reports/data_eval/POSTPROC_EVAL_SPEC.md`; API-контракт —
  `docs/ml_api_contract.md`.

### Ограничения / не измерено
- Реальная LLM не запускалась и не выбиралась (граница задачи).
- Пунктуация — письменный бытовой домен Tatoeba, не спонтанная речь.
- Correction-набор 13 записей; доля улучшенных пар имеет шаг 1/13.
- Покрытие флагов — приближение без лемматизации и позиционного
  сопоставления.


---

## 2026-11-06 — asr-engineer: LLM-постпроцессор QVikhr-3-1.7B int4

### Модель и экспорт
- Выбрана `Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning` (Apache-2.0,
  ревизия `ff7a23e896ebcd28825854b31cee34dfdaedd232`), база Qwen3-1.7B.
- Экспорт: ONNX Runtime GenAI model builder, `qwen3`, **int4** (block_size=32,
  accuracy_level=4), execution_provider=cpu. Скрипт `models/llm/export_qvikhr_int4.py`.
- Размер: BF16 3.44 ГБ → int4 **1.108 ГБ** (1.032 GiB), сжатие ~3.1x.
- Артефакты: `models/llm/QVikhr-3-1.7B-Instruction-noreasoning-int4/`,
  `MANIFEST.md`, `LICENSE` (Apache-2.0), `NOTICE`. Пустой каталог int8 удалён.
- Верификация ONNX int4 vs torch bf16 на 10 примерах: exact 3/10,
  mean char-similarity 0.9856, punct F1 0.6818 vs 0.7619. Суть вывода совпадает.

### Промпт и инструмент
- Промпт: `prompts/postproc_v1.md` (postproc_v1), русский, режимы
  punctuate/correct/full, `/no_think`, JSON-схема для full с конкретным примером.
- Инструмент: `tools/run_llm/run_llm.py` v1.1.0 — добавлены `--backend {torch,onnx}`,
  `--device {cuda,cpu}`, `--metrics`, ручной greedy-цикл с KV-кэшем и замером
  prefill/decode. Запись в `tools/TOOLS.md` обновлена.
- Для GPU установлен torch 2.11.0+cu128 (был CPU-only 2.15.0+cpu);
  transformers 5.22.2, onnxruntime-genai 0.19.0.

### Прогоны (GPU RTX 3080, bf16, greedy)
- punct300: P/R/F1 = 0.8179/0.7039/**0.7566**, boundary F1 **0.8486**;
  1.079 с/предложение, decode 22.15 ток/с, пик CUDA 3.52 ГБ.
- correct13: WER 0.146881 → **0.138833** (delta −0.008048), improved 1,
  worsened 1, unchanged 11; 4.381 с/элемент.
- full63 (clean-50 + correction-13): punct F1 0.4430, boundary F1 0.2029,
  WER 0.146881 → **0.179074** (delta +0.032193), false-alarm **0.16**,
  error coverage 3/119 = 0.0252; 4.268 с/элемент.
- ONNX int4 CPU: 1.515 с/элемент, 17.76 ток/с (10 примеров).

### Что не получилось / ограничения
- `full`-режим (JSON + флаги) ухудшает WER и даёт 16% false-alarm: модель
  перефразирует и галлюцинирует. Для продукта рекомендованы `punctuate` и
  консервативный `correct`, `full` — экспериментальный.
- `correct` исправил только 1 пару из 13; модель 1.7B слишком консервативна
  для ASR-коррекции.
- ONNX-бэкенд в venv CPU-only; GPU-замеры сделаны на torch, не на ORT GenAI.
- Мобильные цифры — аналитические оценки (нет телефона): LLM ~38 с/мин
  (punctuate), пайплайн ~82 с/мин (RTF ~1.4), пик RAM ~1.5–1.8 ГБ.
- Отчёт: `reports/asr_engineer/POSTPROC_REPORT.md`.

---

## 2026-11-06 — merge-engineer: интеграция LLM-постпроцессора (baseline-0.2)

### Решение по интеграции
- Расширен существующий `tools/run_pipeline/merge.py` до **1.1.0**: добавлены
  `--llm-hyp <jsonl>`, `--llm-model <str>`, `--postproc-version <str>`.
  Без новых флагов поведение и схема остаются **baseline-0.1** (обратная
  совместимость); с `--llm-hyp` пишется **baseline-0.2** с additive-полями
  `segments[].text_raw/doubtful/edits` и `meta.llm_model/postproc_version`.
- Новых инструментов не создавалось (лимит цикла исчерпан); использован
  существующий `run_llm` 1.1.0.
- LLM-текст: `punctuate(correct(raw))`. Готовый `llm_torch_correct13.jsonl`
  (режим `correct`, 13/13 id, вход = `<id>.txt`) не даёт пунктуации, поэтому
  поверх него выполнен один прогон `run_llm --mode punctuate` (torch/CUDA,
  bf16, greedy) → `reports/postproc_eval/llm_torch_correct13_punct.jsonl`
  (13 элементов, 60.28 с, 4.637 с/элемент). `full`-режим не использовался:
  по `POSTPROC_REPORT.md` он ухудшает WER (+0.032) и даёт 16 % false-alarm.
- Склейка по сегментам: слова `<id>.txt` и слова LLM-вывода выравниваются
  `difflib.SequenceMatcher` к конкатенации ASR-сегментных слов; каждое слово
  получает сегмент своего якоря. `edits` — word-level diff `text_raw→text`,
  reason из LLM `edits`, иначе `"punct"`. `doubtful` — bool по span-совпадению.
- Результат: 13 JSON в `output/baseline_llm/`, 84 сегмента, 44 сегмента с
  edits, 63 правки (62 replace, 1 delete), 0 doubtful, 940 символов правок.
  `output/baseline/` (baseline-0.1) не изменялся.
- Проверки: v0.2 `--validate-only` — 13/13 PASS; v0.1 `--validate-only` —
  13/13 PASS; merge без `--llm-*` — 0 отличий SHA256 от `output/baseline`;
  повторный merge в `D:\temp\opencode\merge_llm_check` — 0 отличий SHA256;
  без `--force` — exit 2.
- Ограничения: `reason="punct"` для всех правок (correct/punctuate не отдают
  структурированные edits); `ruls_0007` остался без пунктуации; `text_raw`
  берётся из `<id>.txt` и на 7/13 записей чуть отличается от конкатенации
  сегментных текстов `timestamps.json`; `doubtful` на непустых флагах не
  обкатан; качество LLM — по `POSTPROC_REPORT.md`, не переоценивается.
- Отчёт: `reports/merge_engineer/REPORT_LLM.md`; реестр `tools/TOOLS.md`
  обновлён (версия 1.1.0, changelog одной строкой).

---

## 2026-11-06 — reviewer: независимая проверка LLM-постпроцессора

- Вердикт: **PASS WITH NOTES**; замечаний: **9 некритичных, 0 критичных**.
- Перепроверено: punct300 P/R/F1 0.817875/0.703919/0.756630, boundary F1
  0.848580; correct13 WER 0.146881→0.138833; full63 WER +0.032193,
  false-alarm 0.16; ONNX int4 exact 3/10, char-sim 0.9856.
- Контракт: `output/baseline_llm` 13/13 PASS, `output/baseline` 13/13 PASS;
  повторный merge v0.2 и v0.1 — 0 отличий SHA256; повторный eval — отчёт
  побайтно идентичен.
- Лицензии: Apache-2.0 (QVikhr) и Tatoeba CC-BY-2.0-FR присутствуют;
  GrandMaster2 не проверена (признано в отчётах).
- Замечания: неверная атрибуция 1.108 ГБ весам (факт 1.097 ГБ); необъяснённые
  «650 ток/мин» и строки correct/full в §4.3; смешение MB/MiB; завышен
  последовательный пик RAM; `reason="punct"` для всех edits; строчная буква
  в начале сегмента diar_0001; лицензия GrandMaster2; ONNX проверен на 10.
- Отчёт: `reports/reviewer/REVIEW_LLM.md`.

---

## 2026-11-07 — data-eval: no-LLM baseline, COMPARE, T-Wix

### No-LLM baseline (сырой ASR)
- Собраны гипотезы без LLM: `reports/postproc_eval/no_llm_punct300.jsonl`
  (`text_raw` из `punct_eval`) и `no_llm_correct13.jsonl` (`text_asr` из
  `correction_eval`); прогон `tools/eval_postproc` 1.0.0.
- Punct300: P/R/F1 = 1.000000/0.000000/**0.000000**, boundary F1 0.000000
  (TP=0, FN=689, ref boundaries=335). WER на correction13: 0.146881 →
  0.146881, delta 0.000000, improved/worsened/unchanged = 0/0/13.
- Отчёты: `reports/postproc_eval/no_llm_punct300.json/.txt`,
  `no_llm_correct13.json/.txt`.
- Пересчитан `llm_torch_correct13_punct.jsonl` (correct+punct): WER
  0.146881 → 0.142857, delta −0.004024, improved/worsened/unchanged = 1/3/9;
  пунктуация не измерена (в correction-наборе нет пунктуации).
  Отчёт: `reports/postproc_eval/llm_torch_correct13_punct_eval.json/.txt`.
- Сводная таблица: `reports/postproc_eval/COMPARE.md` (no-LLM | punctuate |
  correct+punct | full; источники чисел указаны построчно).

### T-Wix
- Источник: **https://huggingface.co/datasets/t-tech/T-Wix**, ревизия
  `0059b0967915624069b5b4dc224ebe7ef7c041a6`. Лицензия: **ODC-BY-1.0**
  (README; HF tag `license:odc-by`), с оговоркой про third-party model outputs.
- Скачано в `data/twix/raw/` (8 parquet, ~1.70 ГБ; 499 598 строк).
  Схема: `id` (string), `messages` (list<{content, role}>), `subset`
  (general/reasoning/long_context/english_corpus).
- Подготовлено: `data/twix/train_smoke.jsonl` — ровно 10 000 строк
  (`random.Random(42).sample(range(499598), 10000)`, сортировка по глобальному
  индексу); `data/twix/train_full.jsonl` — 499 598 строк; `stats.json`,
  `samples.json`, `keyword_samples.json`.
- **Решение по парам:** T-Wix — общий SFT-mixture, в нём нет пар
  «ASR→исправленный» и «без пунктуации→с пунктуацией». Обучающие пары
  постпроцессора **не выдумывались**; smoke/full сохраняют исходные SFT-записи.
  Для постпроцессора нужен отдельный датасет с такими парами.
- Кандидаты: `t-tech/T-Wix` (канонический), `concretejungles/T-Wix-instag` и
  `...-graded` (явные derivatives), `bethrezen/twix-think` (не derivative,
  не использован), robotics `*Twix*` (нерелевантны).
- Отчёт: `reports/data_eval/TWIX_REPORT.md`.

---

## 2026-11-07 — asr-engineer: MiniCPM5-2B LoRA на T-Wix + postproc-замеры

### Диск
- До старта: C: 5.82 ГБ, D: 4.01 ГБ. HF_HOME остался `D:\hf_cache` (5.33 ГБ сторонних моделей, не трогался).
- **Удалено `data/twix/raw/` (8 parquet, 1.58 ГБ)** — только после проверки, что `train_smoke.jsonl`, `train_full.jsonl`, `stats.json` уже выгружены. Parquet можно повторно скачать из `t-tech/T-Wix` @ `0059b0967915624069b5b4dc224ebe7ef7c041a6`.
- Веса MiniCPM5-2B bf16 (4.70 ГиБ) скачаны в `models/llm_minicpm5/base`, затем перенесены в `C:\temp\opencode\minicpm5_base`, чтобы освободить D: под чекпоинты. После: C: 1.58 ГБ, D: 5.73 ГБ.
- В локальной копии `tokenizer_config.json` `eos_token` заменён на `<|im_end|>` (id 130073), чтобы `run_llm` останавливался на chat-end; `generation_config.json` уже содержит `[1, 130073]`.

### Обучение (smoke)
- Инструмент: `tools/train_llm/train_llm.py` v1.0.0 (новый, запись в `tools/TOOLS.md`).
- Конфиг: LoRA r=16, alpha=32, dropout=0.05, target_modules q/k/v/o/gate/up/down; lr 2e-4, cosine, warmup 75; seq 512; batch 1; grad_accum 4; 1 эпоха; seed 42; bf16; grad checkpointing; adamw_torch.
- Данные: `data/twix/train_smoke.jsonl` (10 000 строк), 2500 шагов.
- Время: **6518 с = 1.811 ч**; пик CUDA **6.350 ГБ**; train_loss **1.1028** (1.165 → 1.089, mean 1.103, min 0.878, max 1.385, NaN нет).
- Лог: `logs/train_minicpm5_smoke.log`; адаптер: `models/llm_minicpm5/smoke/`; summary: `models/llm_minicpm5/smoke/train_summary.json`.

### Full
- Запущен в фоне на `data/twix/train_full.jsonl` (499 598 строк), тот же конфиг, `--save-steps 0`.
- Лог: `logs/train_minicpm5_full.log`; PID: `logs/train_minicpm5_full.pid`.
- Дошло до **31/124900 шагов**, ~2.57 с/шаг, ETA **~89 ч 06 мин**. Остановлен из-за бюджета и роста pagefile на D: (свободно падало до 1.27 ГБ). Full-адаптер не сохранён, full-tuned метрики не измерены.
- Команда повторного запуска — в `reports/asr_engineer/MINICPM_FT_REPORT.md` §6.

### Оценка (eval_postproc 1.0.0, greedy, prompt postproc_v1)
- Важно: `run_llm` 1.1.0 собирает ChatML вручную без пустого ` thinking`-блока; base MiniCPM5 уходит в рассуждение и даёт 0 символов. Для MiniCPM5 использован wrapper `D:\temp\opencode\run_llm_minicpm.py`, который меняет только `build_prompt` (assistant-префикс из chat-шаблона, `enable_thinking=False`); цикл генерации/eos/JSONL/метрики — код `run_llm`. Промпт `prompts/postproc_v1.md` не изменялся.
- MiniCPM5 base: punct300 P/R/F1 = 0.917431/0.290276/**0.441014**, boundary F1 0.770642; correct13 WER 0.146881→0.146881, Δ 0.000000, 1/1/11.
- MiniCPM5 smoke: punct300 P/R/F1 = 0.827922/0.740203/**0.781609**, boundary F1 0.870662; correct13 WER 0.146881→0.156942, Δ **+0.010060**, 1/3/9.
- Full-tuned: не измерено.
- Отчёт: `reports/asr_engineer/MINICPM_FT_REPORT.md`; `reports/postproc_eval/COMPARE.md` дополнен строками MiniCPM5.

## 2026-11-07 — optimizer (повтор): device_bundle / int4 export

- Экспорт MiniCPM5+LoRA smoke → ORT GenAI int4: `models/llm_minicpm5/smoke_int4` (1.64 ГБ), 76.7 с. Логи: `logs/export_smoke_int4.log/.err`.
- Паритет torch (10 примеров, CUDA bf16): 70.73 с, выход `logs/parity_torch10.jsonl`. ONNX int4 — блокер токенизатора ORT GenAI (`\p{L}`/`\p{N}` regex, затем `invalid string_view position`).
- Освобождение диска (разрешено заданием): удалены `D:\hf_cache\models--ai-forever--ru-en-RoSBERTa` (1.5 ГБ), `D:\hf_cache\blobs` (3.78 ГБ, HF-кэш, пере-скачиваем), `data/twix/train_smoke.jsonl` (72 МБ, пере-скачиваем), скачанные архивы. Причина: builder писал ~5.7–6.8 ГБ промежуточного bf16 + 1.6 ГБ int4 → ENOSPC при 8.2 ГБ свободного места.
- Собран `device_bundle/` (1.62 ГБ): models/{asr,diar,llm}, bin/sherpa (.so), audio, scripts/push_to_device.ps1, scripts/run_stage.ps1, MANIFEST.md.
- ASR/диар/LLM на устройстве не измерены: нет CLI-бинарей sherpa (в android.tar.bz2 только jniLibs), llm_cli не собран, токенизатор ORT GenAI несовместим. Отчёт: `reports/optimizer/DEVICE_BUNDLE.md`.

## 2026-11-07 — profiler: device-запуск доделан, замеры ASR/диар на SM-N9750

- **ASR РАБОТАЕТ на устройстве.** Скачан sherpa-onnx v2.0.1 android-aarch64-termux-static (Apache-2.0). Бинари требуют libc++_shared; NDK r27c libc++ не экспортирует std::__ndk1::__hash_memory → собран шим device_bundle/src/hashshim.cpp → libhashshim.so (LD_PRELOAD). /sdcard noexec → бинари в /data/local/tmp/pipeline/bin/sherpa, модели/аудио на /sdcard/pipeline.
- **Диаризация РАБОТАЕТ.** pyannote-segmentation-3-0 (MIT) скачан; model.onnx/int8 падали в ORT 1.22 (topological sort, If-субграф захватывает initializer) → топосорт+копирование initializer'ов → model.fixed.onnx / model.int8.fixed.onnx. threshold 0.7, FastClustering.
- **Замеры (wall / decode RTF):** ASR ruls_0001 (19.88с) 8.3–8.8с / 0.038–0.040; diar ruls_0001 7.5с / 0.309–0.312; diar_0002 (33.5с) 14.7–14.9с / 0.400–0.404; diar_0003 (36.4с) 13.7с / 0.337–0.341. RAM: ASR ~306 МБ, diar 149–163 МБ. Батарея 21.1 °C. /sdcard 79 ГБ свободно. Bundle на устройстве 103 МБ (LLM не пушился).
- **ASR на diar_0002/0003 — не измерено:** zipformer-ru падает (Reshape) на аудио >~20с, нужна нарезка чанков (в CLI нет).
- **LLM — не измерено:** ORT GenAI 0.19.0 (последняя) не принимает tokenizer.json MiniCPM5; LiteRT-LM v0.19.1 не имеет Android-prebuilt (только macOS/iOS), bazel отсутствует. Оба уровня лестницы заблокированы.
- **Гейт RTF ≤ 1.0:** ASR PASS (decode 0.04, wall 0.44), диар PASS (0.31–0.45). LLM — нет вердикта.
- Отчёт: reports/profiler/DEVICE_REPORT.md; stdout прогонов: reports/profiler/device_runs/.

## 2026-11-07 — profiler: LLM QVikhr int4 на устройстве + ASR chunked

- **LLM заработал на SM-N9750.** Собран `llm_cli` (NDK r27c, ORT GenAI 0.19.0 Android AAR + ONNX Runtime 1.23.0 Android AAR). Блокер: GenAI dlopen'ит `libonnxruntime.so` по имени — его не было в GenAI AAR; добавлен из `onnxruntime-android-1.23.0.aar`. Плюс bionic arm64 TLS underalignment → `-fno-emulated-tls` + `alignas(64) thread_local`; линковка `-nostdlib++` (libc++ из libmat), чтобы исключения не пересекали DSO.
- **QVikhr int4 (1.10 ГБ) запушен** на `/sdcard/pipeline/models/llm/`. Замеры (punctuate, greedy, max_new_tokens=256, 1 cold + 2 warm): ruls_0001 wall 12.77–12.83 с / decode RTF 0.227–0.228 / 15.9–16.0 ток/с / RAM 1417 МБ; diar_0002 wall 20.28–20.38 с / 0.331–0.333 / 15.0–15.1 ток/с / RAM 1480 МБ; diar_0003 wall 12.31–12.47 с / 0.115 / 15.9–16.0 ток/с / RAM 1410 МБ.
- **ASR chunked:** `run_stage.ps1 -Chunked` (нарезка ≤19 с на PC, ASR по чанкам, склейка). diar_0002 wall RTF 0.505–0.519 / decode 0.0380–0.0383; diar_0003 wall 0.472–0.475 / decode 0.0381–0.0390; RAM ~298 МБ.
- **Пайплайн (decode-сумма):** ruls_0001 0.577, diar_0002 0.773, diar_0003 0.492 → PASS. Wall-сумма 1.19–1.56 → FAIL из-за загрузки моделей в каждом процессе.
- **Оценка vs замер LLM:** оценка ~38 с/мин, 10–25 ток/с → замер 6.9–19.9 с/мин, 15.0–16.0 ток/с (лучше оценки).
- **Фолбэк:** QVikhr только для device-замеров; punct F1 0.757 vs MiniCPM5-smoke 0.782 (`reports/postproc_eval/COMPARE.md`); продуктовый выбор LLM открыт.
- Батарея после серии 22.2 °C; /sdcard свободно 78 ГБ.


## 2026-11-07 — asr-engineer: оценка CT-Transformer (sherpa-onnx) как замены LLM-пунктуации

- Скачана модель sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8 (Apache-2.0, 61.7 MiB архив / 72.0 MiB model.int8.onnx) в models/punct/; манифест с URL, ревизией, лицензией, размерами и SHA-256 — models/punct/MANIFEST.md.
- Модель китайская (апстрим ModelScope iic/punc_ct-transformer_zh-cn-common-vocab272727): пунктуация <unk>|_|，|。|？|、, русских токенов в словаре нет. На русском тексте склеивает слова и ставит 。: «привет как дела» → «приветкакдела。».
- Прогон 300 пар punct_eval через tools/eval_postproc: P=1.000000 / R=0.000000 / F1=0.000000, boundary F1=0.000000 (TP=0, FN=689). Скорость 3.03 с на 300 сегментов = 10.10 с/1000 предл., 0.0101 с/сегмент (CPU, int8, 2 потока).
- Baseline QVikhr punctuate: P/R/F1 = 0.8179/0.7039/0.7566, boundary F1 = 0.8486 (reports/postproc_eval/COMPARE.md).
- Вердикт: НЕ годится (F1 0.0 << порога 0.72). Русской/мультиязычной CT-Transformer в sherpa-onnx и HF k2-fsa/* нет; русские punct-модели экосистемы — только NeMo ASR (аудио, не текст).
- Отчёт: reports/asr_engineer/PUNCT_MODEL_REPORT.md; метрики: reports/postproc_eval/punctmodel_punct300.json/.txt, гипотезы ..._hyp.jsonl.


## 2026-11-07 - asr-engineer: RUPunct-small INT8 ONNX на punct_eval (300)

- Скачана `ekhodzitsky/rupunct-small-onnx` (MIT, база RUPunct/RUPunct_small) в `models/punct_ru/`; ревизия `c4a35be7e4a413cdfb08427db75b621b571569c9`, ONNX 29 259 673 B, SHA-256 `B105DA023474D98AA13BA18953AE67B04B17BD0595034BC06030C17536893933`. Манифест: `models/punct_ru/MANIFEST.md`.
- Раннер `reports/asr_engineer/run_rupunct.py` (ONNX Runtime + HF WordPiece, aggregation first, upstream process_token). Кириллица OK, склейки/обрезки слов: 0/300 (word-count совпал). Детерминизм подтверждён (одинаковый SHA-256 гипотез).
- Прогон 300 пар через `tools/eval_postproc`: P=0.860643 / R=0.815675 / F1=0.837556, boundary F1=0.947853 (TP=562 FP=91 FN=127). Скорость 0.642 с на 300 = 2.14 мс/сегмент (CPU, inference), загрузка 4.9 с.
- Baseline QVikhr punctuate: P/R/F1 = 0.8179/0.7039/0.7566, boundary F1 0.8486. RUPunct лучше по всем метрикам; порог замены (F1>=0.72, boundary>=0.80) пройден.
- Вердикт: PASS, рекомендуется замена QVikhr-punctuate на RUPunct-small INT8. Ограничения: только пунктуация+регистр, без кавычек/коррекции ASR/doubtful.
- Отчёт: `reports/asr_engineer/RUPUNCT_REPORT.md`; метрики: `reports/postproc_eval/rupunct_punct300.json/.txt`; гипотезы: `...jsonl`.

## 2026-11-07 — asr-engineer: GigaAM v3 RNN-T vs small-zipformer-ru (data/test)

- Лицензия: GigaAM v3 (MIT) подтверждена по метаданным encoder ONNX (`comment=v3`, `version=3`, `is_giga_am=1`, license URL) и upstream `salute-developers/GigaAM/LICENSE` (MIT). Источник: `DaniilYemelyanenka/rashifruy` commit `baa40b5e893b906762ae420a882ef50db031661d`, папка `.../assets/gigaam/`.
- Скачано 4 файла в `models/gigaam_v3/` (221.15 MiB; encoder.int8 224.6 MB, decoder 4.6 MB, joiner 2.7 MB, tokens 13.4 KB); MANIFEST с URL/SHA-256/лицензией — `models/gigaam_v3/MANIFEST.md`.
- WER (eval_asr, data/test, 13 файлов, 497 слов): zipformer 0.146881 micro / 0.144337 macro (S/D/I 49/14/10); GigaAM v3 0.120724 / 0.101869 (30/27/3). Гипотезы: `tools/eval_asr/hyp/gigaam_v3/`.
- RTF CPU (threads=1, 2 прохода): zipformer 0.018417 (chunk 19 с), GigaAM 0.088437 (chunk 60 с). Пик RAM 338.8 / 473.2 MB.
- GPU: sherpa pip CPU-only; onnxruntime-gpu 1.31.1 — encoder CUDA EP + decoder/joiner CPU + Python greedy: RTF 0.081581, пик VRAM 3231 MB. Кросс-WER GPU-декода 0.100604.
- Гейт (WER ≤ 0.146881 и RTF ≤ 0.5): PASS. Рекомендация: качество лучше (−17.8% relative), но размер ×8.6 (221 vs ~25.7 MiB) и больше удалений на длинных diar_*; не слепая замена, нужен A/B на устройстве.
- Отчёт: `reports/asr_engineer/GIGAAM_VS_ZIPFORMER.md`; WER-отчёты `reports/asr_engineer/wer_{zipformer,gigaam_v3}.{json,txt}`.

## 2026-11-07 22:36 - orchestrator: чистка проекта
- Удалено (воспроизводимое/дубликаты, ~7.7 ГБ): android/app/build (4.4 ГБ, APK остаётся в android/out/Meevo.apk), android/out/* кроме Meevo.apk, device_bundle/models + audio (дубликаты models/), models/llm_minicpm5/smoke_int4 (экспорт, восстановим tools/export_llm; adapter smoke/ СОХРАНЁН), models/llm_src, models/llm_build_cache, models/punct (провальный CT-Transformer, см. PUNCT_MODEL_REPORT.md), logs/*.pid.
- Сохранено: все рабочие модели (gigaam_v3, llm/QVikhr, punct_ru, diar, small-zipformer), data/, tools/, reports/, output/, промпты, исходники android/, train_smoke.jsonl (воспроизводимость обучения).
