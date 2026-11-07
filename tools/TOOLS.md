# Реестр инструментов

Единственный источник правды по инструментам проекта. Перед созданием нового
инструмента проверь этот файл. Дубликаты запрещены.

Запуск всех инструментов — интерпретатором проекта:
`D:\Downloads_D\nuclear_hack\.venv\Scripts\python.exe`.

---

### prep_audio
- Назначение: приведение аудио к моно WAV с целевой частотой дискретизации (по умолчанию 16 кГц) и проверка длительности.
- Когда использовать: подготовка сырых записей к eval/ASR; конвертация mp3/flac/ogg/иных WAV в единый формат; проверка, что файл читается и имеет ненулевую длительность.
- Когда НЕ использовать: нарезка на сегменты по VAD/спикерам (это отдельная задача), денойзинг/нормализация громкости, конвертация в формат, отличный от WAV.
- Запуск:
  ```
  .venv\Scripts\python.exe tools\prep_audio\prep_audio.py data\raw\extracted data\test --sample-rate 16000
  .venv\Scripts\python.exe tools\prep_audio\prep_audio.py input.wav out_dir --force
  ```
- Вход → выход: вход — файл или папка (нерекурсивно) с аудио; выход — папка с `<stem>.wav` (PCM16, моно, целевой sample rate). Печатает версии soundfile/numpy, длительность каждого файла и суммарную.
- Автор/агент: data-eval; версия 1.0.0; дата 2026-11-06; проверен на: `data/test` (13 файлов, 310.641 с).
- Известные ограничения: ресемплинг — линейная интерполяция (достаточно для eval, не studio-grade); нет ffmpeg-фолбэка, поддержка форматов ограничена libsndfile (wav/flac/ogg/mp3 и совместимые); папка обрабатывается нерекурсивно; без `--force` не перезаписывает существующие выходные файлы (код возврата 2).

### eval_asr
- Назначение: word-level WER по манифесту с русской нормализацией (нижний регистр, ё→е, удаление пунктуации, схлопывание пробелов), отчёт по каждой записи и суммарно.
- Когда использовать: сравнение ASR-кандидатов на одном наборе; проверка гипотез `<id>.txt`; быстрый self-test (гипотеза = эталон → WER 0).
- Когда НЕ использовать: оценка диаризации/DER (см. eval_diar); оценка качества без эталонных транскрипций; языки, где нужна другая нормализация (скрипт ориентирован на RU).
- Запуск:
  ```
  .venv\Scripts\python.exe tools\eval_asr\eval_asr.py --manifest data\test\manifest.jsonl --hyp hyp_dir --report reports\data_eval\asr_report
  .venv\Scripts\python.exe tools\eval_asr\eval_asr.py --manifest data\test\manifest.jsonl --hyp-json hyp.json --report reports\data_eval\asr_report
  ```
- Вход → выход: вход — `manifest.jsonl` (id, path, text_path, duration_sec, source, license), папка `<id>.txt` / один `.txt` на все записи / JSON `id→text`; выход — `<report>.json` + `<report>.txt` (per-record WER, S/D/I, суммарный micro-WER и macro-WER). Печатает версию Python.
- Автор/агент: data-eval; версия 1.0.0; дата 2026-11-06; проверен на: `data/test` (13 записей, 497 слов; ref→WER 0; 1 замена→WER 0.002012 micro).
- Известные ограничения: WER считается по словам после нормализации; пунктуация и регистр не оцениваются; нет опции игнорировать вставки/удаления отдельно; один `.txt` в `--hyp` применяется ко всем записям (режим для быстрых проверок).

### eval_diar
- Назначение: frame-based DER по RTTM (ref vs hyp) с collar 0.25 с, разбивкой по числу спикеров и по перекрытиям.
- Когда использовать: оценка диаризации на записях с 2+ говорящими; проверка RTTM-гипотез; self-test ref vs ref → DER 0.
- Когда НЕ использовать: оценка ASR/WER (см. eval_asr); разметка аудио (инструмент только считает метрику); сценарии без RTTM-референса.
- Запуск:
  ```
  .venv\Scripts\python.exe tools\eval_diar\eval_diar.py --ref data\test\diar_0001.rttm --hyp hyp\diar_0001.rttm --collar 0.25 --report reports\data_eval\der_report
  .venv\Scripts\python.exe tools\eval_diar\eval_diar.py --ref ref_dir --hyp hyp_dir --collar 0.25 --report reports\data_eval\der_report
  ```
- Вход → выход: вход — RTTM-файл или папки `*.rttm` с совпадающими именами; выход — `<report>.json` + `<report>.txt` (per-file DER, miss/FA/speaker error, overlap/non-overlap, by ref speaker count). Печатает версию Python.
- Автор/агент: data-eval; версия 1.0.1; дата 2026-11-06; проверен на: `data/test` (3 RTTM, 2/3/4 спикера; ref→DER 0; удалён первый сегмент→DER 0.065819) и синтетическом overlap-тесте (ref→DER 0; пропуск перекрывающегося спикера→overlap DER 0.5).
- Известные ограничения: шаг кадра 10 мс; collar применяется к границам референса; speaker mapping — brute force для ≤7 гипотезных спикеров, иначе greedy; DER считается по speaker-time (перекрытия учитываются по каждому активному спикеру); RTTM-сегменты синтетического набора не содержат реальных пересечений, поэтому overlap-ветка проверена отдельным handcrafted-тестом.

### run_diar
- Назначение: offline-диаризация через sherpa-onnx (pyannote segmentation-3.0 → эмбеддинги спикера → FastClustering) с записью RTTM-гипотез и JSON-отчёта (wall time, RTF, число спикеров).
- Когда использовать: baseline/сравнение конфигураций диаризации на манифесте; генерация `<id>.rttm` для `eval_diar`; проверка ложных спикеров на односпикеровых записях.
- Когда НЕ использовать: оценка DER (см. `eval_diar`); ASR/слияние; квантизация/экспорт под Android (optimizer); сценарии без 16 кГц моно WAV.
- Запуск:
  ```
  .venv\Scripts\python.exe tools\run_diar\run_diar.py --manifest data\test\manifest.jsonl --out reports\diar_engineer\hyp\A --threshold 0.7 --report reports\diar_engineer\run_A.json --force
  .venv\Scripts\python.exe tools\run_diar\run_diar.py --manifest data\test\manifest.jsonl --out reports\diar_engineer\hyp\B --oracle-num-speakers --report reports\diar_engineer\run_B.json --force
  ```
- Вход → выход: вход — `manifest.jsonl` (id, path, duration_sec, speakers_count), папка моделей; выход — `<out>/<id>.rttm` + JSON-отчёт (per-file duration/wall/RTF/num_speakers_hyp/ref, суммарный RTF, версии). Печатает прогресс по каждой записи.
- Автор/агент: diar-engineer; версия 1.0.0; дата 2026-11-06; проверен на: `data/test` (13 записей; A threshold 0.7 → DER 0.480088 на 3 diar, 0/10 ложных спикеров; B oracle → DER 0.551991).
- Известные ограничения: только 16 кГц моно WAV; модели задаются путями (по умолчанию `models/diar/...`); `--oracle-num-speakers` берёт `speakers_count` из manifest (не «слепой» режим); FastClustering может вернуть меньше кластеров, чем задано в `num_clusters`; RTF — Python/CPU, не замер Android; без `--force` не перезаписывает RTTM (код 2).

### run_asr
- Назначение: прогон offline ASR-бэкенда по манифесту с сохранением гипотез, таймингов и метрик (wall-time, RTF, пик RSS).
- Когда использовать: получить `<id>.txt` для `eval_asr`; снять token/word/segment тайминги; замерить RTF и пик RAM на CPU для сравнения ASR-кандидатов.
- Когда НЕ использовать: оценка WER (см. `eval_asr`); диаризация/слияние спикеров; замер на Android (нужен `bench_device`/profiler); streaming-инференс.
- Запуск:
  ```
  .venv\Scripts\python.exe tools\run_asr\run_asr.py --model models\sherpa-onnx-zipformer-ru-2024-09-18 --manifest data\test\manifest.jsonl --out tools\eval_asr\hyp\zipformer_ru --backend sherpa-onnx --precision int8 --threads 1 --repeat 2 --max-chunk-sec 20 --force
  ```
- Вход → выход: вход — каталог модели (encoder/decoder/joiner + tokens.txt), `manifest.jsonl`, `--backend` (сейчас только `sherpa-onnx`), `--precision int8|fp32`, `--threads`, `--repeat`, `--max-chunk-sec`; выход — `<out>/<id>.txt`, `<out>/timestamps.json` (tokens/words/segments), `<out>/metrics.json` (passes, avg, RTF, peak_rss_mb, per-record).
- Автор/агент: asr-engineer; версия 1.0.0; дата 2026-11-06; проверен на: `data/test` (13 записей, 310.641 с; A WER 0.175050, B WER 0.146881; повторный прогон A — 0 отличий).
- Известные ограничения: только backend `sherpa-onnx` (offline transducer); zipformer-ru-2024-09-18 падает на входе >~20 с, поэтому по умолчанию нарезка по 20 с без перекрытия (границы могут терять слова); `words[].end` — время начала последнего токена, не конца звучания; `segments[]` — эвристика по паузе ≥ `--segment-gap` (0.5 с), не разметка модели; пик RSS включает Python + onnxruntime, не равен бюджету Android; без `--force` не перезаписывает существующие `.txt` (код возврата 2).

### run_pipeline (merge.py)
- Назначение: слияние ASR-сегментов с говорящими из RTTM по пересечению времени и выдача JSON по OUTPUT CONTRACT; опциональная интеграция LLM-постпроцессинга (additive-поля `text_raw`/`doubtful`/`edits`, `meta.llm_model`/`meta.postproc_version`).
- Когда использовать: есть готовые `timestamps.json` ASR и `<id>.rttm` диаризации; нужен `output/baseline/<id>.json`; проверка JSON на соответствие контракту (`--validate` / `--validate-only`); интеграция per-record LLM-вывода в `output/baseline_llm`.
- Когда НЕ использовать: запуск ASR/диаризации (см. `run_asr`, `run_diar`); оценка WER/DER (см. `eval_asr`, `eval_diar`); изменение моделей/порогов; прогон LLM (см. `run_llm`).
- Запуск:
  ```
  .venv\Scripts\python.exe tools\run_pipeline\merge.py --asr-hyp tools\eval_asr\hyp\small_zipformer_ru --rttm reports\diar_engineer\hyp\A_campplus_thr0.7 --manifest data\test\manifest.jsonl --out output\baseline --validate
  .venv\Scripts\python.exe tools\run_pipeline\merge.py --asr-hyp tools\eval_asr\hyp\small_zipformer_ru --rttm reports\diar_engineer\hyp\A_campplus_thr0.7 --manifest data\test\manifest.jsonl --out output\baseline_llm --llm-hyp reports\postproc_eval\llm_torch_correct13_punct.jsonl --llm-model "Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning int4" --postproc-version "postproc_v1 (punctuate+correct)" --validate
  .venv\Scripts\python.exe tools\run_pipeline\merge.py --out output\baseline --validate-only
  ```
- Вход → выход: вход — папка ASR с `timestamps.json` и `<id>.txt`, папка RTTM `<id>.rttm`, `manifest.jsonl`; опционально `--llm-hyp` (JSONL `id,text,doubtful?,edits?`), `--llm-model`, `--postproc-version`; выход — `<out>/<id>.json` по OUTPUT CONTRACT (v0.1) или с additive-полями (v0.2) + сводка статистики в stdout. `--validate` проверяет после слияния, `--validate-only` — только существующие файлы (обе версии схемы).
- Автор/агент: merge-engineer; версия 1.1.0; дата 2026-11-06; проверен на: `data/test` (13 записей, 84 сегмента; v0.1 13/13 valid, повторный прогон — 0 отличий SHA256; v0.2 13/13 valid, 44 сегмента с edits, 63 edits, 0 doubtful, повторный прогон — 0 отличий SHA256).
- Известные ограничения: назначение спикера — только по времени (перекрытия перебиваний не восстанавливаются, т.к. в ASR-таймингах нет меток перекрытия); `confidence` = 0.0 («не измерено», модель не даёт уверенности); нулевые по длине ASR-сегменты расширяются на 0.01 с для выполнения `start < end`; без `--force` не перезаписывает существующие JSON (код возврата 2); LLM-текст распределяется по сегментам выравниванием слов (difflib) к ASR-сегментам, поэтому границы предложений могут попадать на границу сегмента; reason для правок — из LLM `edits`, иначе `"punct"` (режим `correct` не отдаёт структурированные edits).
- Changelog: 1.1.0 (2026-11-06) — добавлены `--llm-hyp/--llm-model/--postproc-version`, additive-поля v0.2 и их валидация; поведение без новых флагов не изменилось (v0.1).

### eval_postproc
- Назначение: оценка LLM-постпроцессора русских ASR-транскрипций — восстановление пунктуации (P/R/F1 по знакам и границам предложений), дельта WER до/после коррекции, качество флагов «сомнительное словосочетание» (false-alarm proxy + покрытие известных ASR-ошибок).
- Когда использовать: сравнение LLM-режимов `punctuate`/`correct`/`flagDoubtful` на фиксированном наборе; проверка, что коррекция не ухудшает WER; self-test (hyp = ref → punct F1 = 1.0, delta WER = 0).
- Когда НЕ использовать: оценка ASR без постпроцессинга (см. `eval_asr`); оценка диаризации (см. `eval_diar`); обучение/выбор LLM; языки, отличные от русского.
- Запуск:
  ```
  .venv\Scripts\python.exe tools\eval_postproc\eval_postproc.py --hyp hyp.jsonl --ref-punct data\postproc\punct_eval.jsonl --ref-corr data\postproc\correction_eval.jsonl --report reports\postproc_eval\report.json
  .venv\Scripts\python.exe tools\eval_postproc\eval_postproc.py --hyp reports\postproc_eval\selftest_ref.jsonl --ref-punct data\postproc\punct_eval.jsonl --ref-corr data\postproc\correction_eval.jsonl --report reports\postproc_eval\selftest_ref.json --force
  ```
- Вход → выход: вход — `--hyp` JSONL (`{"id","text","doubtful"?,"edits"?}`) или каталог `<id>.txt`; `--ref-punct` `punct_eval.jsonl`; `--ref-corr` `correction_eval.jsonl`; выход — `<report>.json` + `<report>.txt` (micro/macro punct P/R/F1, boundary P/R/F1, WER before/after/delta, improved/worsened/unchanged, false-alarm rate, flag density, error coverage, per-sentence/per-pair детали). Печатает версию Python; сторонних зависимостей нет.
- Автор/агент: data-eval; версия 1.0.0; дата 2026-11-06; проверен на: `data/postproc` (300 punct-пар, 13 correction-пар) + self-test (`selftest_ref`: punct F1 1.000000, delta WER 0.000000; `selftest_no_commas`: F1 0.675000; `selftest_flags`: false-alarm 0.100000, error coverage 0.092437).
- Известные ограничения: пунктуация оценивается только по набору знаков `.,!?;:—–…` (кавычки/скобки/дефисы внутри слов не учитываются); выравнивание слов — `difflib.SequenceMatcher`, при сильном расхождении текстов часть знаков уходит в FP/FN; WER-нормализация как в `eval_asr` (пунктуация и регистр не учитываются); покрытие флагов считается по surface form без лемматизации и позиционного сопоставления; correction-набор — 13 записей (грубая доля улучшенных пар); без `--force` не перезаписывает отчёты (код возврата 2).

### run_llm
- Назначение: детерминированный (greedy, temperature=0) LLM-постпроцессор русских ASR-транскрипций: пунктуация/регистр, исправление ошибок ASR, флаг `doubtful` + варианты замены; два бэкенда — `torch` (transformers, CUDA bf16 / CPU fp32) и `onnx` (ONNX Runtime GenAI, CPU).
- Когда использовать: прогон LLM на `data/postproc/*.jsonl` или каталоге `<id>.txt`; сравнение режимов `punctuate`/`correct`/`full`; проверка ONNX-комплекта против torch-версии; снятие скорости (prefill/decode токены/с, сек/элемент, пик CUDA-памяти).
- Когда НЕ использовать: оценка метрик (см. `eval_postproc`); обучение/выбор LLM; ASR/диаризация/слияние; языки, отличные от русского; ONNX-бэкенд на GPU (в текущей сборке только `CPUExecutionProvider`).
- Запуск:
  ```
  .venv\Scripts\python.exe tools\run_llm\run_llm.py --model C:\temp\opencode\qvikhr_src --in data\postproc\punct_eval.jsonl --out reports\postproc_eval\llm_punct.jsonl --mode punctuate --backend torch --device cuda --metrics reports\postproc_eval\llm_punct_metrics.json --force
  .venv\Scripts\python.exe tools\run_llm\run_llm.py --model models\llm\QVikhr-3-1.7B-Instruction-noreasoning-int4 --in data\postproc\punct_eval.jsonl --out reports\postproc_eval\llm_punct_onnx.jsonl --mode punctuate --backend onnx --device cpu --force
  ```
- Вход → выход: вход — `--model` (HF id или локальный каталог transformers для `torch`; каталог с `genai_config.json` для `onnx`), `--in` (`jsonl` с `text`/`text_raw`/`text_asr`, `.txt` или каталог `<id>.txt`), `--mode {punctuate,correct,full}`, `--backend {torch,onnx}`, `--device {cuda,cpu}`, `--max-new-tokens`, `--threads`, `--prompt-templates` (по умолчанию `prompts/postproc_v1.md`), `--metrics`; выход — JSONL `{"id","text","doubtful","edits":[{"from","to","reason"}]}` (или каталог `<id>.txt`); опционально JSON-метрики (load/wall, сек/элемент, prompt/generated токены, prefill/decode с и токены/с, пик CUDA-памяти, per-item).
- Автор/агент: asr-engineer; версия 1.1.0; дата 2026-11-06; проверен на: `data/postproc/punct_eval.jsonl` (300, torch CUDA bf16), `correction_eval.jsonl` (13, correct), clean-50 + correction-13 (full), ONNX int4 на 10 примерах (сверка с torch: exact 3/10, mean char-similarity 0.9856).
- Известные ограничения: `torch`-бэкенд требует CUDA-сборку torch для `--device cuda` (в проекте стоит 2.11.0+cu128); `onnx`-бэкенд в текущей сборке CPU-only; `full`-режим заметно ухудшает WER на correction-наборе (модель склонна перефразировать/галлюцинировать) — для текста предпочтительны `punctuate`/`correct`; `--max-new-tokens` дополнительно ограничивается эвристикой от длины промпта (защита от продолжения оборванного текста); без `--force` не перезаписывает выход (код возврата 2); качество зависит от промпта в `prompts/postproc_v1.md`.

### train_llm
- Назначение: LoRA SFT-обучение MiniCPM5-2B (и совместимых `LlamaForCausalLM` chat-моделей) на JSONL с `messages`; assistant-only loss через chat-шаблон модели, bf16, детерминированный seed, прогресс Trainer/tqdm в лог.
- Когда использовать: дообучение LLM на SFT-миксе (T-Wix) на одной GPU; smoke/full LoRA-прогоны; воспроизводимый лог обучения с версиями и loss history.
- Когда НЕ использовать: QLoRA/bitsandbytes (в этой сборке не используется), full fine-tune (не влезает в 10 ГБ), оценка качества (см. `eval_postproc`), инференс (см. `run_llm`).
- Запуск:
  ```
  .venv\Scripts\python.exe tools\train_llm\train_llm.py --model C:\temp\opencode\minicpm5_base --data data\twix\train_smoke.jsonl --out models\llm_minicpm5\smoke --epochs 1 --lr 2e-4 --lora-r 16 --lora-alpha 32 --max-seq-len 512 --batch 1 --grad-accum 4 --save-steps 0 --log logs\train_minicpm5_smoke.log --force
  ```
- Вход → выход: вход — `--model` (HF id или локальный каталог), `--data` JSONL (`{"messages":[{"role","content"}]}`), `--out` каталог; выход — LoRA-адаптер (`adapter_config.json`, `adapter_model.safetensors`), tokenizer, `train_summary.json` (конфиг, версии, loss history, время, пик CUDA). Прогресс Trainer/tqdm пишется в `--log`.
- Автор/агент: asr-engineer; версия 1.0.0; дата 2026-11-07; проверен на: `data/twix/train_smoke.jsonl` (10 000 строк, 2500 шагов, 1.811 ч, train_loss 1.1028, пик CUDA 6.350 ГБ, RTX 3080).
- Известные ограничения: только LoRA bf16 (QLoRA/bitsandbytes на Windows не используется); assistant-only loss — по маркерам chat-шаблона, без `{% generation %}`-патча; `--max-seq-len` обрезает длинные примеры; `--save-steps 0` отключает промежуточные чекпоинты (сохраняется только финальный адаптер); `--resume` требует `checkpoint-*` в `--out`; полный T-Wix (499 598 строк) на RTX 3080 ≈ 89 ч (1 эпоха, seq 512, batch 1, grad-accum 4); без `--force` не перезаписывает непустой `--out` (код 2).
