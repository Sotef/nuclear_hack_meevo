# data-eval: тестовый набор и инструменты оценки

Дата: 2026-11-06
Python: 3.12.10 (`.venv\Scripts\python.exe`)
Окружение: Windows, `D:\Downloads_D\nuclear_hack`

## 1. Итог

| Показатель | Значение |
|---|---|
| Записей в `data/test` | 13 |
| Суммарная длительность | 310.64 с (5.18 мин) |
| Формат | WAV, PCM16, моно, 16 кГц |
| Эталонные транскрипции | 13/13 (`<id>.txt`) |
| Манифест | `data/test/manifest.jsonl` (13 строк, валиден) |
| Записей с 2+ говорящими | 3 (синтетические, 2/3/4 спикера, есть RTTM) |
| Реальных человеческих записей с 2+ говорящими и транскрипцией | не найдено (см. §5) |
| Лицензии | CC-BY-4.0 (10 файлов), MIT (3 файла) |

## 2. Состав набора

| id | Длит., с | Слов | Источник | Лицензия | Спикеров |
|---|---:|---:|---|---|---:|
| ruls_0001 | 19.880 | 30 | istupakov/russian_librispeech (OpenSLR SLR96, LibriVox) | CC-BY-4.0 | 1 |
| ruls_0002 | 19.790 | 34 | istupakov/russian_librispeech (OpenSLR SLR96, LibriVox) | CC-BY-4.0 | 1 |
| ruls_0003 | 19.760 | 32 | istupakov/russian_librispeech (OpenSLR SLR96, LibriVox) | CC-BY-4.0 | 1 |
| ruls_0004 | 19.580 | 35 | istupakov/russian_librispeech (OpenSLR SLR96, LibriVox) | CC-BY-4.0 | 1 |
| ruls_0005 | 19.520 | 44 | istupakov/russian_librispeech (OpenSLR SLR96, LibriVox) | CC-BY-4.0 | 1 |
| ruls_0006 | 19.460 | 34 | istupakov/russian_librispeech (OpenSLR SLR96, LibriVox) | CC-BY-4.0 | 1 |
| ruls_0007 | 19.390 | 42 | istupakov/russian_librispeech (OpenSLR SLR96, LibriVox) | CC-BY-4.0 | 1 |
| ruls_0008 | 19.140 | 28 | istupakov/russian_librispeech (OpenSLR SLR96, LibriVox) | CC-BY-4.0 | 1 |
| ruls_0009 | 19.060 | 30 | istupakov/russian_librispeech (OpenSLR SLR96, LibriVox) | CC-BY-4.0 | 1 |
| ruls_0010 | 18.980 | 36 | istupakov/russian_librispeech (OpenSLR SLR96, LibriVox) | CC-BY-4.0 | 1 |
| diar_0001 | 46.132 | 58 | ivkond/synthetic-speech-diarization-ru | MIT | 2 |
| diar_0002 | 33.507 | 52 | ivkond/synthetic-speech-diarization-ru | MIT | 3 |
| diar_0003 | 36.441 | 42 | ivkond/synthetic-speech-diarization-ru | MIT | 4 |

Статистика длительностей: min 18.980 с, mean 23.895 с, max 46.132 с.
Слов после RU-нормализации: 497 (совпадает с эталоном в self-test).

### Источники и лицензии (проверено по карточкам)

- **istupakov/russian_librispeech** — HF dataset card: `license: cc-by-4.0`; исходный корпус RuLS (OpenSLR SLR96) собран из аудиокниг LibriVox, аудио Public Domain в США. Использование с атрибуцией допустимо.
- **ivkond/synthetic-speech-diarization-ru** — HF dataset card: `license: mit`; 2000 синтетических треков 16 кГц с сегментами спикеров и текстом. MIT допускает использование.
- **Golos** не взят: в карточке `bond005/sberdevices_golos_10h_crowd` лицензия указана как `other` / «Public license with attribution and conditions reserved», а не CC-BY-4.0; плюс минимальный parquet-шард ~98–400 МБ. Лицензию не стали трактовать расширительно.
- **Mozilla Common Voice ru** не взят: официальный датасет gated; доступные HF-зеркала в webdataset-формате имеют test-шард 924 МБ (Sh1man/common_voice_21_ru), что избыточно для 3–10 минут.

## 3. Структура

```
data/raw/                 # скачанные parquet + извлечённые исходники
data/raw/extracted/       # 13 wav + txt + rttm + meta.json (до конвертации)
data/test/                # 13 wav (16k/mono/PCM16) + 13 txt + 3 rttm + manifest.jsonl
tools/prep_audio/         # prep_audio.py
tools/eval_asr/           # eval_asr.py
tools/eval_diar/          # eval_diar.py
tools/TOOLS.md            # реестр инструментов
reports/data_eval/        # REPORT.md, отчёты self-test, гипотезы
logs/journal.md           # журнал решений
```

## 4. Самопроверка инструментов

### prep_audio 1.0.0

- `--help` работает, аргументы `input output --sample-rate --force` описаны.
- Прогон на `data/raw/extracted` → `data/test`: 13/13 OK, суммарно 310.641 с, все 16 кГц/моно/PCM16.
- Повторный запуск без `--force`: код возврата **2**, сообщение «output file(s) already exist; refusing to overwrite without --force» и список файлов.
- Повторный запуск с `--force`: код возврата **0**, 13/13 OK.

### eval_asr 1.0.0

- `--help` работает.
- Гипотеза = эталон (`reports/data_eval/hyp_ref/`): **WER = 0.000000** (497 слов, 0 ошибок), отчёт `asr_ref.json/.txt`.
- Одна замена в `ruls_0001` (`reports/data_eval/hyp_one_error/`): **WER = 0.002012** micro (1 ошибка / 497 слов), WER записи `ruls_0001` = **0.033333** (1/30), отчёт `asr_one_error.json/.txt`.
- `--hyp-json` с эталонными текстами: **WER = 0.000000**, отчёт `asr_json_ref.json/.txt`.
- Режим одного `.txt` на все записи: запускается, WER 0.921529 (ожидаемо бессмысленная гипотеза), отчёт `asr_single_file.json/.txt`.
- Несуществующий путь гипотез: код возврата **2**, понятная ошибка.

### eval_diar 1.0.1

- `--help` работает.
- 3 RTTM из `data/test` (2/3/4 спикера), ref vs ref: **DER = 0.000000**, 9040 ref speaker-time кадров (10 мс; перекрытия считаются по каждому активному спикеру), отчёт `der_all_ref.json/.txt`.
- Те же 3 файла, из гипотезы удалён первый сегмент каждого: **DER = 0.065819** (miss=595 speaker-time кадров), отчёт `der_all_perturbed.json/.txt`.
- Handcrafted overlap-тест (`reports/data_eval/overlap_test/`): ref vs ref **DER = 0.000000**, 98 overlap speaker-time кадров; удалён перекрывающийся спикер → **overlap DER = 0.500000**, overall DER = 0.284058, отчёт `der_overlap_missing.json/.txt`.

## 5. Baseline-метрики и честные «не измерено»

| Метрика | Значение | Комментарий |
|---|---|---|
| WER ASR-модели | **не измерено** | ASR-модели не выбирались и не запускались (граница задачи: asr-engineer). Инструмент проверен на эталонных/искусственных гипотезах. |
| DER на реальных записях | **не измерено, нет референса** | Для 10 реальных записей RuLS есть только транскрипция, без RTTM-разметки спикеров. |
| DER на синтетических записях | **0.000000** (ref vs ref) | Это self-test инструмента, а не качество диаризации. |
| Реальные человеческие записи с 2+ говорящими + транскрипция | **не найдено** | В рамках бюджета не найден открытый русский набор с реальной многоговорящей речью, транскрипцией и однозначной лицензией. Найдены синтетические 2/3/4-спикерные треки (MIT), они и включены. |

## 6. Ограничения набора

- 10 реальных записей — чтение поэтических аудиокниг (Пушкин), один диктор на запись; это не спонтанная разговорная речь и не телефонный канал.
- 3 многоговорящие записи — синтез речи (TTS), а не живые люди; RTTM-сегменты в них последовательные, без реальных пересечений (флаг `has_overlaps=True` относится к полю `simultaneous_segments`, но в `speakers` пересечений нет). Overlap-ветка eval_diar проверена отдельным handcrafted-тестом.
- Нет записей с шумом/дальним микрофоном/телефонным каналом; для реалистичного baseline это стоит добавить позже.
- Ресемплинг в prep_audio — линейная интерполяция; для исходников 16 кГц ресемплинг не выполнялся.
- В eval_diar `ref_frames` / `ref_speaker_frames` — это speaker-time кадры (10 мс): при перекрытии кадр учитывается для каждого активного спикера, поэтому сумма может превышать число физических кадров.

## 7. Воспроизведение

```powershell
# данные уже в data/test; пересборка из сырья:
.venv\Scripts\python.exe data\raw\extract_data.py
.venv\Scripts\python.exe tools\prep_audio\prep_audio.py data\raw\extracted data\test --sample-rate 16000 --force

# WER self-test
.venv\Scripts\python.exe tools\eval_asr\eval_asr.py --manifest data\test\manifest.jsonl --hyp reports\data_eval\hyp_ref --report reports\data_eval\asr_ref

# DER self-test
.venv\Scripts\python.exe tools\eval_diar\eval_diar.py --ref data\test --hyp reports\data_eval\hyp_diar_ref --collar 0.25 --report reports\data_eval\der_all_ref
```

---

## 8. LLM-постпроцессор: eval-набор и харнесс (2026-11-06)

### 8.1 Что собрано

| Артефакт | Содержимое |
|---|---|
| `data/postproc/punct_eval.jsonl` | **300** пар «текст без пунктуации → референс с пунктуацией»; 50 помечены `clean=true` |
| `data/postproc/correction_eval.jsonl` | **13** пар «ASR-гипотеза → эталон» (497 слов) |
| `data/postproc/build_stats.json` | статистика сборки (seed, число кандидатов, лицензии) |
| `data/postproc/build_postproc_data.py` | детерминированная сборка (seed=42) |
| `tools/eval_postproc/eval_postproc.py` | харнесс 1.0.0 (единственный новый инструмент цикла) |
| `reports/data_eval/POSTPROC_EVAL_SPEC.md` | формулы метрик, формат входа, ограничения, воспроизведение |
| `docs/ml_api_contract.md` | lifecycle, поля запроса, additive-поля, errorCode, стадии прогресса |

**Источники и лицензии (проверено):**

- Пунктуация: **Tatoeba Russian sentences** (`rus_sentences.tsv.bz2`,
  https://downloads.tatoeba.org/exports/per_language/rus/rus_sentences.tsv.bz2).
  Tatoeba для текстовых предложений по умолчанию применяет **CC-BY 2.0 FR**
  (атрибуция; коммерческое использование разрешено). Из 315 600 подходящих
  предложений детерминированно (`seed=42`) отобрано 300.
- Коррекция: существующие артефакты проекта — `data/test/<id>.txt`
  (CC-BY-4.0 / MIT по `manifest.jsonl`) и
  `tools/eval_asr/hyp/small_zipformer_ru/<id>.txt` (baseline WER 0.146881).
- UD ru-SynTagRus **не взят**: в `LICENSE.txt` репозитория указана
  **CC BY-NC-SA 4.0** (NonCommercial), что не подходит для использования в
  приложении. Это зафиксировано в журнале решений.

### 8.2 Self-test `eval_postproc` 1.0.0

| Сценарий | Результат |
|---|---|
| `selftest_ref` (punct hyp = ref, correction hyp = ASR) | punct micro **P=1.000000 R=1.000000 F1=1.000000** (TP=689, FP=0, FN=0, 300 предложений); boundary **F1=1.000000** (335/335); WER before=after=**0.146881**, **delta=0.000000**, unchanged=13/13 |
| `selftest_corrected` (punct hyp = ref, correction hyp = ref) | punct F1=**1.000000**; WER before=0.146881 → after=**0.000000**, **delta=-0.146881**, improved=12/13, unchanged=1/13 |
| `selftest_no_commas` (удалены запятые) | punct micro **P=1.000000 R=0.509434 F1=0.675000** (TP=351, FN=338); boundary F1=1.000000; delta WER=0.000000 |
| `selftest_flags` (5 флагов на clean + флаги на ASR-ошибках) | false-alarm rate=**0.100000** (5/50), плотность 1.089 флага/100 слов; покрытие ошибок **11/119 = 0.092437**, пар покрыто 11/12 |
| `--hyp` = каталог `<id>.txt` | работает (2 записи, punct F1=1.000000) |
| повторный запуск без `--force` | код возврата **2**, понятное сообщение |
| несуществующий `--hyp` | код возврата **2**, понятное сообщение |

Отчёты: `reports/postproc_eval/selftest_ref.json/.txt`,
`selftest_corrected.json/.txt`, `selftest_no_commas.json/.txt`,
`selftest_flags.json/.txt`, `selftest_dir.json/.txt`.
Канонический пример выхода: `reports/postproc_eval/report.json/.txt`
(это прогон `selftest_ref`, не результат реальной LLM).

### 8.3 Метрики, которые умеет `eval_postproc`

- Пунктуация: micro/macro precision/recall/F1 по знакам
  `.,!?;:—–…`; precision/recall/F1 по границам предложений.
- Коррекция: WER before/after, delta WER (micro и macro), число и доля
  улучшенных/ухудшенных/неизменных пар.
- Флаги: false-alarm rate и плотность флагов на 50 чистых предложениях;
  покрытие известных ASR-ошибок (surface-form, без лемматизации) и покрытие
  пар.
- Версии: Python и версия инструмента; сторонних зависимостей нет.

### 8.4 «Не измерено» / ограничения

- **Качество реальной LLM не измерено**: в этом цикле LLM не запускалась и не
  выбиралась (граница задачи — asr-engineer/model-scout). Все цифры выше —
  self-test инструмента на эталонных/искусственных гипотезах.
- Пунктуация оценивается на письменных предложениях Tatoeba (бытовой домен),
  а не на спонтанной речи/телефонном канале.
- В `data/test/*.txt` нет пунктуации, поэтому `correction_eval` не может
  оценивать пунктуацию; для неё отдельный `punct_eval`.
- Correction-набор — 13 записей (497 слов); доля улучшенных пар имеет шаг 1/13.
- Покрытие флагов — приближение по surface form; позиционное сопоставление
  флага с конкретным словом ASR-гипотезы не выполняется.
- Лицензия Tatoeba требует атрибуции; при поставке приложения нужно сохранить
  notices (Tatoeba contributors, CC-BY 2.0 FR).
