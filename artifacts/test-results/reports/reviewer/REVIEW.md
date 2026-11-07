# reviewer: независимая проверка baseline-цикла

Дата: 2026-11-06. Автор: reviewer.
Окружение: Windows, `D:\Downloads_D\nuclear_hack`, Python 3.12.10 (`.venv\Scripts\python.exe`).
Режим: только чтение артефактов проекта; проверочные прогоны писались во временные пути
`D:\temp\opencode\review_reports` и `D:\temp\opencode\review_merge`. Модели ASR/диаризации
заново не запускались — только eval-инструменты на готовых гипотезах.

## 1. Воспроизводимость метрик

### 1.1 WER (eval_asr 1.0.0, готовые гипотезы)

Команды:

```powershell
.venv\Scripts\python.exe tools\eval_asr\eval_asr.py --manifest data\test\manifest.jsonl `
  --hyp tools\eval_asr\hyp\small_zipformer_ru --report D:\temp\opencode\review_reports\asr_small
.venv\Scripts\python.exe tools\eval_asr\eval_asr.py --manifest data\test\manifest.jsonl `
  --hyp tools\eval_asr\hyp\zipformer_ru --report D:\temp\opencode\review_reports\asr_big
```

| Кандидат | Заявлено (REPORT.md) | Перепроверено | S/D/I | Совпало |
|---|---:|---:|---|---|
| B: small-zipformer-ru | WER micro 0.146881 | **0.146881** (73/497) | 49/14/10 | да |
| A: zipformer-ru | WER micro 0.175050 | **0.175050** (87/497) | 43/34/10 | да |
| B: macro | 0.144337 | **0.144337** | — | да |
| A: macro | 0.169596 | **0.169596** | — | да |

Per-record WER также совпал с таблицей `reports/asr_engineer/REPORT.md` §3 (все 13 строк,
включая ruls_0002 A=0.4412, diar_0003 A=0.3571 / B=0.2857).

### 1.2 DER (eval_diar 1.0.1, конфиг A)

Команда:

```powershell
.venv\Scripts\python.exe tools\eval_diar\eval_diar.py --ref data\test `
  --hyp reports\diar_engineer\hyp\A_campplus_thr0.7 --collar 0.25 `
  --report D:\temp\opencode\review_reports\der_A
```

| Метрика | Заявлено | Перепроверено | Совпало |
|---|---:|---:|---|
| OVERALL DER | 0.480088 | **0.480088** | да |
| miss / fa / spk_err | 3052 / 95 / 1193 | **3052 / 95 / 1193** | да |
| ref_speaker_frames | 9040 | **9040** | да |
| diar_0001 DER | 0.500528 | **0.5005** | да |
| diar_0002 DER | 0.351160 | **0.3512** | да |
| diar_0003 DER | 0.572536 | **0.5725** | да |
| OVERLAP DER | не измерено (0 ref-кадров) | **0.000000 при ref_speaker_frames=0** | да (подтверждено) |

### 1.3 Ложные спикеры на односпикеровых записях (FA 0/10)

Проверено независимо по RTTM-гипотезам `reports/diar_engineer/hyp/A_campplus_thr0.7/ruls_*.rttm`:
во всех 10 файлах единственный уникальный спикер — `SPEAKER_1`. Заявленное **0/10** подтверждено.

### 1.4 Валидация OUTPUT CONTRACT

```powershell
.venv\Scripts\python.exe tools\run_pipeline\merge.py --out output\baseline --validate-only
```

Результат: `[validate] PASS` для всех 13 файлов, итог **13/13 files valid**, код возврата 0.
Заявленное 13/13 подтверждено.

### 1.5 Детерминизм повторного слияния

```powershell
.venv\Scripts\python.exe tools\run_pipeline\merge.py `
  --asr-hyp tools\eval_asr\hyp\small_zipformer_ru `
  --rttm reports\diar_engineer\hyp\A_campplus_thr0.7 `
  --manifest data\test\manifest.jsonl --out D:\temp\opencode\review_merge --validate
```

Сводка прогона совпала с отчётом merge-engineer: `files=13 segments=84 no_overlap=2
small_overlap=3 repaired=5`, `13/13 files valid`.
Сравнение SHA256 всех 13 JSON с `output/baseline`: **files=13, diffs=0**.
Заявленный детерминизм подтверждён.

## 2. Полнота отчётов (EVIDENCE RULES)

Требуемый минимум: набор данных, версия модели, параметры, метрики, ограничения.

| Отчёт | Набор данных | Версия модели | Параметры | Метрики | Ограничения | Итог |
|---|---|---|---|---|---|---|
| `reports/data_eval/REPORT.md` | есть (13 записей, источники, лицензии) | н/п (модели не оценивались; указаны версии инструментов) | есть (prep_audio/eval_asr/eval_diar, collar, шаг кадра) | есть (self-test WER/DER) | есть (§6) | полный |
| `reports/model_scout/CANDIDATES.md` | н/п (разведка, без эксперимента) | есть (теги/версии кандидатов) | н/п (модели не запускались) | **нет собственных метрик — заявлено явно** («без экспериментального WER»); опубликованные WER с источниками | есть (лицензионные риски, открытые вопросы) | полный для роли scout |
| `reports/asr_engineer/REPORT.md` | есть (data/test, 310.641 с, 497 слов) | есть (2024-09-18, sherpa-onnx 2.0.1) | есть (int8, threads=1, repeat=2, max-chunk=20, greedy, dither=0) | есть (WER micro/macro, S/D/I, RTF, RAM, размер) | есть (§5, §7 «не измерено») | полный |
| `reports/diar_engineer/REPORT.md` | есть (3 diar + 10 ruls) | есть (pyannote seg-3.0, CAM++, TitaNet, sherpa-onnx 2.0.1) | есть (threshold 0.7, min_duration_on/off, collar 0.25, 1 поток) | есть (DER, miss/FA/spk_err, RTF, FA 0/10) | есть (§8) | полный |
| `reports/merge_engineer/REPORT.md` | есть (13 записей, входные артефакты) | есть (ASR small-zipformer int8, diar A, merge 0.1) | есть (алгоритм overlap, tie-break, fallback, округление) | есть (84 сегмента, 2 fallback, 3 малых, 5 repaired, 13/13 valid) | есть (§5) | полный |

Замечаний по полноте, блокирующих приёмку, нет. Отсутствие собственных метрик в
`CANDIDATES.md` корректно оговорено и соответствует роли model-scout.

## 3. Лицензии

Сверка манифестов с карточками (выборочно, ключевые):

| Объект | Заявлено в манифесте | Проверено | Итог |
|---|---|---|---|
| Silero VAD | MIT (`CANDIDATES.md` §2) | `raw.githubusercontent.com/snakers4/silero-vad/master/LICENSE` → **MIT License, © 2020-present Silero Team** | подтверждено; риск устаревшего бейджа CC BY-NC в README остаётся, но LICENSE = MIT |
| RuLS (`istupakov/russian_librispeech`) | CC-BY-4.0 | HF dataset card → **License: cc-by-4.0**; описание: Public Domain in USA (LibriVox) | подтверждено |
| `ivkond/synthetic-speech-diarization-ru` | MIT | HF dataset card → **License: mit** | подтверждено |
| pyannote segmentation-3.0 | MIT | HF model card → **License: mit**; доступ к карточке gated (нужно принять условия) | подтверждено; gated-доступ — организационное, не лицензионное ограничение |
| CAM++ (3D-Speaker) | Apache-2.0 | по манифесту/`CANDIDATES.md`; карточку весов ModelScope в этом ревью не открывал | **не проверено независимо** (причина: не открывал карточку весов) |
| TitaNet-small | CC-BY-4.0 | по манифесту/`CANDIDATES.md`; карточку не открывал | **не проверено независимо** |

Сомнительные/требующие внимания:
- **Silero VAD**: LICENSE = MIT, но в README исторически висел бейдж CC BY-NC 4.0.
  Для продукта фиксировать MIT по LICENSE; при юридической строгости — письменное подтверждение.
- **pyannote segmentation-3.0**: MIT, но HF-репозиторий gated. Использованный ONNX-пакет
  взят из GitHub release `k2-fsa/sherpa-onnx`; в архиве есть LICENSE (MIT, © 2022 CNRS).
- **CAM++ / TitaNet**: лицензии взяты из `CANDIDATES.md`/манифеста, независимо не сверены
  с карточками весов. Для продукта нужен отдельный лицензионный аудит весов.
- **Датасеты**: обе заявленные лицензии (CC-BY-4.0, MIT) подтверждены карточками HF.

## 4. OUTPUT CONTRACT (выборочная проверка)

Проверены глазами + через `--validate-only`:

- `output/baseline/ruls_0007.json` (односпикеровый): top-level ключи `audio_id`,
  `duration_sec`, `language`, `speakers`, `segments`, `meta` — совпадают со схемой;
  `language="ru"`, `duration_sec=19.39` (совпадает с manifest), `speakers=[{id:"SPEAKER_1",
  label:null}]`, сегменты упорядочены по `start` (0.0 → 8.88), `start < end`,
  `confidence=0.0`, `meta` содержит ровно `asr_model/diar_model/version`.
- `output/baseline/diar_0003.json` (многоговорящий): 5 спикеров (`SPEAKER_1..4, SPEAKER_6`),
  8 сегментов, порядок по `start` (0.0 → 32.64), все `speaker` входят в `speakers`,
  `start < end`, `confidence=0.0`, `meta` по схеме. Пропуск `SPEAKER_5` — следствие
  нумерации RTTM, контрактом не запрещён.

Оба файла проходят `--validate-only` (13/13 PASS). Отклонений от OUTPUT CONTRACT не найдено.

## 5. Вердикт

**PASS WITH NOTES.**

Все заявленные числа воспроизведены независимо и точно: WER обоих кандидатов, DER конфига A,
FA 0/10, 13/13 валидных JSON, детерминизм повторного слияния (0 отличий SHA256).
Критичных замечаний нет. Ниже — некритичные.

### Замечания

**Критичные:** нет.

**Некритичные:**

1. `reports/model_scout/CANDIDATES.md` §2 / `models/diar/MANIFEST.md`: лицензия Silero VAD
   заявлена как MIT — LICENSE подтверждает MIT, но в README репозитория остаётся бейдж
   CC BY-NC 4.0. Нужно зафиксировать MIT в лицензионном аудите и, при строгом требовании,
   получить письменное подтверждение.
2. `models/diar/MANIFEST.md`: лицензии весов CAM++ (Apache-2.0) и TitaNet-small (CC-BY-4.0)
   взяты из `CANDIDATES.md` и не сверены с карточками весов в этом ревью — «не проверено
   независимо», требуется отдельный аудит весов.
3. `models/diar/MANIFEST.md`: pyannote segmentation-3.0 — MIT, но HF-карточка gated;
   использован ONNX-пакет из GitHub release sherpa-onnx. Для продукта зафиксировать
   источник и LICENSE из архива (в архиве LICENSE есть).
4. `reports/asr_engineer/REPORT.md` §1 / `models/MANIFEST.md`: «размер int8-комплекта»
   (66.8 / 25.7 MiB) посчитан с `decoder.int8.onnx`, но в рантайме используется
   `decoder.onnx` (fp32, ~2.09 МБ). Фактический размер рантайм-комплекта чуть больше
   заявленного int8; на выводы не влияет, но формулировку стоит уточнить.
5. `reports/diar_engineer/REPORT.md` §1: overlap-DER «не измерено» при
   `ref_speaker_frames=0` — перепроверено, инструмент возвращает 0.000000 при нулевом
   знаменателе. Формулировка корректна, но значение 0.0 не должно трактоваться как
   качество на перекрытиях.
6. `data/test/manifest.jsonl` / `reports/data_eval/REPORT.md`: у 3 diar-записей стоит
   `has_overlaps: true`, при этом в RTTM 0 overlap-кадров. В отчёте это объяснено
   (`has_overlaps` относится к `simultaneous_segments`), но поле манифеста может вводить
   в заблуждение потребителей — стоит добавить комментарий/поле `rttm_overlap_frames=0`.

## 6. Команды проверки (сводка)

```powershell
# WER x2
.venv\Scripts\python.exe tools\eval_asr\eval_asr.py --manifest data\test\manifest.jsonl --hyp tools\eval_asr\hyp\small_zipformer_ru --report D:\temp\opencode\review_reports\asr_small
.venv\Scripts\python.exe tools\eval_asr\eval_asr.py --manifest data\test\manifest.jsonl --hyp tools\eval_asr\hyp\zipformer_ru --report D:\temp\opencode\review_reports\asr_big
# DER
.venv\Scripts\python.exe tools\eval_diar\eval_diar.py --ref data\test --hyp reports\diar_engineer\hyp\A_campplus_thr0.7 --collar 0.25 --report D:\temp\opencode\review_reports\der_A
# Контракт
.venv\Scripts\python.exe tools\run_pipeline\merge.py --out output\baseline --validate-only
# Детерминизм (во временную папку)
.venv\Scripts\python.exe tools\run_pipeline\merge.py --asr-hyp tools\eval_asr\hyp\small_zipformer_ru --rttm reports\diar_engineer\hyp\A_campplus_thr0.7 --manifest data\test\manifest.jsonl --out D:\temp\opencode\review_merge --validate
# SHA256: 13 файлов, 0 отличий от output\baseline
```
