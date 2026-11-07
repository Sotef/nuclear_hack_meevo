# Кандидаты моделей: ASR (RU), VAD, диаризация

Дата: 2026-11-06. Автор: model-scout.
Статус: разведка (без скачивания моделей и без экспериментального WER).

Ограничения отбора:
- целевой рантайм: ONNX Runtime Mobile, sherpa-onnx, whisper.cpp, TFLite/LiteRT;
- Android arm64-v8a, телефон среднего класса;
- язык по умолчанию — русский;
- лицензия должна разрешать использование в приложении (MIT / Apache-2.0 / BSD / CC-BY и т.п.);
- NC / закрытые / gated-лицензии помечены отдельно.

Обозначения:
- **«не указано»** — значение отсутствует в проверенном источнике; не оценивалось.
- **«оценка»** — расчёт из числа параметров (fp32 ≈ 4 байта/параметр, int8 ≈ 1 байт/параметр), не измерение.
- WER — только опубликованные значения с указанием источника; собственных замеров нет.
- Размеры sherpa-моделей — сумма файлов из листинга `ls -lh` в документации sherpa-onnx.

---

## 1. ASR (русский)

| # | Имя (версия/тег) | Источник URL | Размер, МБ | Языки | Лицензия / статус | Формат | Рантайм | Тайминги слов/сегментов | Streaming | Ожидаемое качество на RU (опубликовано) | Примечания |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `sherpa-onnx-zipformer-ru-2024-09-18` (int8) | https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html ; донор: https://huggingface.co/alphacep/vosk-model-ru | ≈67 (encoder.int8 65 + decoder.int8 1.2 + joiner.int8 0.25 + tokens/bpe 0.25); fp32 ≈250 | ru-only | **Apache-2.0** (Vosk model-ru; https://alphacephei.com/vosk/models) — разрешено | ONNX (transducer: encoder/decoder/joiner) | sherpa-onnx (в т.ч. Android APK) | Да, token-level timestamps в JSON | Нет (offline; можно «simulated streaming» с VAD) | Vosk model-ru-0.42: 4.5 audiobooks, 11.1 open_stt audiobooks, 19.5 open_stt youtube, 36.0 openstt calls, 4.4 golos crowd, 17.9 sova devices (https://alphacephei.com/vosk/models) | Лучший «готовый» baseline: int8, ~67 МБ, официальный Android-пример sherpa. Донор — Vosk/Kaldi, сконвертирован в ONNX. Обновлённая версия `sherpa-onnx-zipformer-ru-2025-04-20` существует, размер в карточке не указан. |
| 2 | `sherpa-onnx-small-zipformer-ru-2024-09-18` (int8) | https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html ; донор: https://huggingface.co/alphacep/vosk-model-small-ru | ≈26 (encoder.int8 24 + decoder.int8 1.2 + joiner.int8 0.25 + tokens/bpe 0.25); fp32 ≈89 | ru-only | **Apache-2.0** (Vosk model-small-ru; https://alphacephei.com/vosk/models) — разрешено | ONNX (transducer) | sherpa-onnx | Да, token-level | Нет (offline) | Vosk model-small-ru-0.22: 22.71 openstt audiobooks, 31.97 openstt youtube, 29.89 sova devices, 11.79 golos crowd (https://alphacephei.com/vosk/models) | Самый лёгкий RU-вариант для «слабых» телефонов; качество заметно ниже большой модели. |
| 3 | `sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19` (int8) | https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html ; донор: https://github.com/salute-developers/GigaAM | ≈231 (encoder.int8 226 + decoder 3.2 + joiner 1.4) | ru-only | **GigaAM License NC** (non-commercial) — **ЗАПРЕЩЕНО** для приложения; см. https://github.com/salute-developers/GigaAM/blob/main/GigaAM%20License_NC.pdf | ONNX (NeMo transducer) | sherpa-onnx | Да, token-level | Нет (offline; simulated streaming) | GigaAM-RNNT v2 (243M): call-center 10.22, other telephony 7.88, CV19 5.32, OpenSTT asr_calls_2_val 20.07 (https://huggingface.co/t-tech/T-one) | Сильная RU-модель, но лицензия NC → только для исследований/сравнения, не для продукта. |
| 4 | `GigaAM-v3` (`ctc` / `rnnt` / `e2e_ctc` / `e2e_rnnt`) | https://huggingface.co/ai-sage/GigaAM-v3 | 220–240M параметров; ≈900 fp32 / ≈250 int8 (**оценка**) | ru + en | **MIT** (карточка HF) — разрешено | PyTorch (официально); ONNX — через сторонний экспорт (напр. `istupakov/onnx-asr`) | ONNX Runtime (после экспорта); официального sherpa-onnx-пакета нет | Да (CTC frame-level / RNNT) | Нет (offline) | V3 CTC / RNNT: Open datasets 3.0 / 2.6; Golos Farfield 4.5 / 3.9; Natural Speech 7.8 / 6.9; Callcenter 10.3 / 9.5; Average 9.2 / 8.4 (https://huggingface.co/ai-sage/GigaAM-v3) | Лучшее опубликованное качество на RU среди permissive-моделей. Подводные камни: нужен собственный ONNX-экспорт и проверка квантизации; 220–240M параметров тяжелее Zipformer-small. |
| 5 | `t-tech/T-one` (71M) | https://huggingface.co/t-tech/T-one ; код: https://github.com/voicekit-team/T-one | 71.7M параметров; ≈287 fp32 / ≈72 int8 (**оценка**) | ru-only | **Apache-2.0** — разрешено | PyTorch (safetensors) + ONNX (HF tag `onnx`; есть community ONNX) | ONNX Runtime; в sherpa-onnx официально нет | Да, word-level timestamps в pipeline | **Да** (chunks 300 мс) | Call-center 8.63; other telephony 6.20; named entities 5.83; CV19 5.32; OpenSTT asr_calls_2_val original 20.27 / re-labeled 7.94 (https://huggingface.co/t-tech/T-one) | Лучший кандидат для телефонии/стриминга. Подводные камни: нужен KenLM-декодер и phrase splitter; интеграция в Android сложнее, чем sherpa-ONNX. |
| 6 | `nvidia/stt_ru_conformer_transducer_large` (120M) | https://huggingface.co/nvidia/stt_ru_conformer_transducer_large | 120M параметров; ≈480 fp32 / ≈120 int8 (**оценка**) | ru-only | **CC-BY-4.0** — разрешено с атрибуцией | NeMo `.nemo` (PyTorch); ONNX — через скрипты sherpa-onnx | ONNX Runtime / sherpa-onnx (после конвертации) | Да (RNNT) | Нет (offline) | MCV10 dev 3.5 / test 4.0; Golos-crowd 2.7; Golos-farfield 7.6; RuLS 12.0 (https://huggingface.co/nvidia/stt_ru_conformer_transducer_large) | Хорошее качество, но требуется конвертация NeMo→ONNX; проверять лицензии датасетов (Golos, CV, RuLS, SOVA). |
| 7 | `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` (25 европейских языков, включая ru) | https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html ; донор: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 | ≈640 (encoder.int8 622 + decoder 12 + joiner 6.1) | multi (25 языков, ru есть) | **CC-BY-4.0** (NVIDIA NeMo) — разрешено с атрибуцией | ONNX (NeMo TDT transducer) | sherpa-onnx (есть Android APK) | Да, token-level | Нет (offline; simulated streaming) | WER именно на RU в карточке не опубликован — **не указано** | Мультиязычный «тяжёлый» вариант: 640 МБ int8 может быть много для среднего телефона; зато один файл на 25 языков. |
| 8 | `whisper.cpp` + `ggml-large-v3-turbo` / `ggml-small` / `ggml-base` (multilingual) | https://github.com/ggml-org/whisper.cpp ; веса: https://huggingface.co/ggerganov/whisper.cpp | small 466 MiB; base 142 MiB; large 2.9 GiB; **turbo в README не указан** | multi (включая ru) | **MIT** (whisper.cpp и веса OpenAI Whisper) — разрешено | ggml (bin) | whisper.cpp (Android-пример есть) | Да: сегменты; word-level через `-ml 1` | Псевдо (`whisper-stream`, шаг 0.5 с) | Whisper (large-v3) в таблице GigaAM-v3: Open datasets 12.0; Golos Farfield 16.7; Natural Speech 13.6; Callcenter 23.9; Average 25.1 (https://huggingface.co/ai-sage/GigaAM-v3) | Простая интеграция, но на RU заметно хуже RU-специфичных моделей; large не влезает в бюджет среднего телефона; известны галлюцинации на тишине/шуме. |
| 9 | `vosk-model-small-ru-0.22` / `vosk-model-ru-0.42` | https://alphacephei.com/vosk/models | 45 / 1800 | ru-only | **Apache-2.0** — разрешено | Kaldi (не ONNX) | Vosk/Kaldi — **не входит** в разрешённый список рантаймов | Да (word-level в Vosk API) | Да (Vosk streaming API) | small-ru-0.22: 22.71 / 31.97 / 29.89 / 11.79; ru-0.42: 4.5 / 11.1 / 19.5 / 36.0 / 4.4 / 17.9 (https://alphacephei.com/vosk/models) | Использовать только как донора для sherpa-ONNX (п.1–2). Прямой Vosk-runtime нарушает ограничение «только ONNX/whisper.cpp/TFLite». |
| 10 | `Moonshine` (moonshine-ai) | https://github.com/moonshine-ai/moonshine | tiny от 1 МБ; base — не указано | **русского нет** (STT: en, es, de, zh, ja, ko, vi, tl, uk, ar) | **MIT** (модели MIT, кроме legacy non-English — NC) — разрешено, но неприменимо | ONNX / собственный runtime | on-device (Android есть) | Да | Да | **неприменимо**: RU не поддерживается | Включён как «отклонённый кандидат»: русский язык отсутствует в списке STT. |
| 11 | `Silero STT` (silero-models) | https://github.com/snakers4/silero-models | не указано | ru (заявлен) | **не подтверждена** в этом ревью; по репозиторию — CC BY-NC 4.0 (NC) → **запрещено до подтверждения** | PyTorch / ONNX-экспорт официально не заявлен | ONNX Runtime (только при самостоятельном экспорте) | не указано | не указано | не указано | Не рекомендуется для baseline: лицензия NC и отсутствие официального ONNX-пакета. Требует отдельной юридической проверки. |

### Что важно по ASR
- Самый «дешёвый» вход в baseline — п.1 (≈67 МБ int8, Apache-2.0, официальный sherpa-onnx Android).
- Если нужен streaming/телефония — п.5 (T-one, Apache-2.0, 300 мс chunks), но интеграция сложнее.
- Если приоритет качество и допустим размер ~250 МБ int8 — п.4 (GigaAM-v3, MIT), но нужен собственный ONNX-экспорт.
- П.3 (GigaAM v2 в sherpa) — только для сравнения, в продукт нельзя (NC).

---

## 2. VAD

| # | Имя (версия) | Размер | Лицензия / статус | Формат | Рантайм | Streaming | Примечания |
|---|---|---|---|---|---|---|---|
| 1 | **Silero VAD v6.2.0** | ≈2 МБ (JIT; ONNX-файл ~2 МБ) | **MIT** (LICENSE репозитория) — разрешено; см. риск ниже | ONNX; также ggml (whisper.cpp) | ONNX Runtime, sherpa-onnx, whisper.cpp (`--vad`) | Да (окна 30+ мс; 512 samples @16k) | Основной baseline. 8/16 кГц, обучен на 6000+ языков, <1 мс на чанк на 1 потоке (https://github.com/snakers4/silero-vad). В sherpa-onnx поставляется как `silero_vad.onnx`. |
| 2 | **Silero VAD v5.1.2 / v4.0** | ≈2 МБ | **MIT** — разрешено | ONNX | ONNX Runtime, sherpa-onnx | Да | Более старые версии; v4 — окно 30 мс, v5 — 512 samples. Выбор версии фиксировать в конфиге (разные пороги/поведение). |
| 3 | **WebRTC VAD** | не указано (код библиотеки, без отдельного файла модели) | **BSD-3-Clause** (WebRTC) — разрешено | C++ (GMM, не ONNX) | нативная библиотека (не входит в список ONNX/whisper.cpp/TFLite) | Да (10/20/30 мс кадры) | Очень лёгкий, но качество в шуме хуже Silero; нет модели как файла. Использовать только если нужен минимальный бинарный размер. |
| 4 | **TEN VAD** (поддержан в sherpa-onnx как `ten_vad`) | не указано | **не подтверждена** в этом ревью (по репозиторию TEN-framework/ten-vad — Apache-2.0) | ONNX | sherpa-onnx | Да (окно 256 samples) | Альтернатива Silero; в sherpa-onnx есть `TenVadModelConfig`. Перед использованием — юридическая проверка лицензии. |
| 5 | **whisper.cpp VAD (Silero ggml)** | ≈0.86 МБ (`ggml-silero-v6.2.0.bin`) | **MIT** — разрешено | ggml | whisper.cpp | Да | Тот же Silero, но в ggml; удобно, если ASR уже на whisper.cpp. |

### Что важно по VAD
- Silero VAD — де-факто стандарт: MIT, ~2 МБ, ONNX, streaming, уже интегрирован и в sherpa-onnx, и в whisper.cpp.
- WebRTC VAD — только как fallback при жёстком лимите на размер/зависимости.

---

## 3. Диаризация (офлайн, RU)

| # | Имя (версия) | Размер | Лицензия / статус | Формат | Рантайм | Clustering | Примечания |
|---|---|---|---|---|---|---|---|
| 1 | **pyannote segmentation-3.0** (ONNX-конвертация sherpa) | 5.7 МБ fp32 / 1.5 МБ int8 | **MIT** (pyannote/segmentation-3.0; в sherpa-пакете есть LICENSE) — разрешено | ONNX | sherpa-onnx | Нет (только сегментация; кластеризация отдельно) | Лучший baseline-сегментатор. В sherpa-onnx: `sherpa-onnx-pyannote-segmentation-3-0`. Требует модель эмбеддингов + кластеризацию. |
| 2 | **3D-Speaker CAM++** (`speech_campplus_sv_zh-cn_16k-common`) | не указано (ONNX в sherpa-onnx release) | **Apache-2.0** (3D-Speaker) — разрешено | ONNX | sherpa-onnx | Да, через sherpa-onnx `FastClustering` (threshold или `num_clusters`) | Эмбеддинги спикера; EER 0.65% VoxCeleb1-O, 6.78% CNCeleb, 7.75% 3D-Speaker (https://github.com/modelscope/3D-Speaker). Обучен на китайском, но эмбеддинги слабо зависят от языка; для RU нужен замер DER. |
| 3 | **3D-Speaker ERes2Net-base** | не указано | **Apache-2.0** — разрешено | ONNX | sherpa-onnx | Да (FastClustering) | EER 0.84% VoxCeleb1-O; 6.61M параметров. Альтернатива CAM++ при лучшем EER на некоторых доменах. |
| 4 | **NVIDIA NeMo TitaNet-small** (`nemo_en_titanet_small.onnx`) | не указано | **CC-BY-4.0** (NVIDIA NeMo) — разрешено с атрибуцией | ONNX | sherpa-onnx | Да (FastClustering) | В примере sherpa-onnx даёт RTF 0.11–0.12 на 56.9 с аудио (CPU). Обучен на английском; для RU — только через замер DER. |
| 5 | **WeSpeaker** (ResNet34 / ECAPA-TDNN, ONNX) | не указано | **Apache-2.0** (wenet-e2e/wespeaker) — разрешено | ONNX (runtime/onnxruntime) | ONNX Runtime; часть моделей есть в sherpa-onnx release | Да (собственный pipeline: UMAP+HDBSCAN или PLDA) | Хорошая библиотека эмбеддингов; для Android нужно брать ONNX-экспорт и писать кластеризацию/склейку самостоятельно. |
| 6 | **pyannote/speaker-diarization-3.1** (полный pipeline) | не указано | **MIT**, но **gated** (нужно принять условия HF) — формально разрешено, практически ограничено | PyTorch (в 3.1 убран onnxruntime) | PyTorch — **недопустим** в приложении | Да (встроенная) | **Нельзя** использовать напрямую на Android: PyTorch-зависимость и gated-доступ. Для офлайн-приложения брать только segmentation-3.0 (п.1) + эмбеддинги. |
| 7 | **sherpa-onnx-reverb-diarization-v1** | 9.1 МБ fp32 / 2.3 МБ int8 | **Non-commercial** (Revai reverb-diarization-v1) — **ЗАПРЕЩЕНО** | ONNX | sherpa-onnx | Да | Явно NC; в продукт нельзя. |
| 8 | **3D-Speaker full diarization pipeline** | не указано | **Apache-2.0** — разрешено | PyTorch (Python) | PyTorch — недопустим в приложении | Да (VAD + segmentation + embedding + clustering) | Использовать только как референс/для экспорта эмбеддингов; в Android — только ONNX-части. |

### Что важно по диаризации
- Рабочая связка для Android: **pyannote segmentation-3.0 (ONNX, 1.5 МБ int8) + 3D-Speaker CAM++ (ONNX) + sherpa-onnx FastClustering**.
- Clustering в sherpa-onnx умеет работать без знания числа спикеров (`--clustering.cluster-threshold`); если число известно — `--clustering.num-clusters`.
- pyannote 3.1 целиком — не вариант для Android (PyTorch + gated).

---

## Рекомендация для baseline

### ASR
**Основной:** `sherpa-onnx-zipformer-ru-2024-09-18` (int8, ≈67 МБ, Apache-2.0).
Почему: готовый ONNX под sherpa-onnx, официальный Android-пример, token-level timestamps, приемлемый WER на Golos/аудиокнигах (4.4–11.1), малый размер.
Подводные камни: offline-модель (не настоящий streaming), донор — Vosk/Kaldi (версия донора в карточке не зафиксирована), на телефонии WER высокий (36.0 на openstt calls).

**Альтернатива для streaming/телефонии:** `t-tech/T-one` (Apache-2.0, 71M, 300 мс chunks, word-level timestamps).
Подводные камни: нужен KenLM-декодер/phrase splitter, нет готового sherpa-onnx-пакета, интеграция в Android сложнее.

**Если приоритет качество:** `GigaAM-v3` (MIT, 220–240M, avg WER 8.4–9.2).
Подводные камни: официально PyTorch; нужен собственный ONNX-экспорт и проверка int8; размер ~250 МБ int8.

### VAD
**Silero VAD v6.2.0** (ONNX, ≈2 МБ, MIT).
Почему: уже поддерживается sherpa-onnx и whisper.cpp, streaming, 8/16 кГц, минимальный размер.
Подводные камни: в README репозитория есть устаревший бейдж CC BY-NC 4.0 — юридически ориентироваться на LICENSE (MIT), но зафиксировать это в лицензионном аудите.

### Диаризация
**pyannote segmentation-3.0 (int8, 1.5 МБ) + 3D-Speaker CAM++ (ONNX) + sherpa-onnx FastClustering.**
Почему: всё permissive (MIT + Apache-2.0), работает офлайн на ONNX Runtime, clustering не требует заранее знать число спикеров (threshold), есть готовый Android-пример sherpa-onnx.
Подводные камни: CAM++ обучен на китайском — DER на русском нужно измерить; качество на перебиваниях и шуме не гарантировано; число спикеров лучше ограничивать (`min/max` или threshold).

---

## Лицензионные риски

### Запрещено / нельзя в продукт
| Модель | Лицензия | Почему |
|---|---|---|
| `sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19` | GigaAM License NC | Non-commercial; в sherpa-карточке явно «for non-commercial use only». |
| `sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24` | GigaAM License NC | То же; в пакете лежит `GigaAM License_NC.pdf`. |
| `sherpa-onnx-reverb-diarization-v1` | Non-commercial (Revai) | Явно NC в документации sherpa. |
| `Silero STT` (silero-models) | CC BY-NC 4.0 (по репозиторию; **требует подтверждения**) | NC → нельзя без коммерческой лицензии. |
| `pyannote/speaker-diarization-3.1` (полный pipeline) | MIT, но gated + PyTorch | Формально MIT, но gated-доступ и PyTorch-зависимость нарушают runtime-ограничения. |

### Сомнительно / требует юридической проверки
| Модель | Что проверить |
|---|---|
| Silero VAD | В README бейдж «CC BY-NC 4.0», а LICENSE и текст README — MIT. Зафиксировать MIT в аудите; при сомнении — запросить письменное подтверждение у Silero. |
| TEN VAD | Лицензия не подтверждена в этом ревью; проверить репозиторий TEN-framework/ten-vad перед использованием. |
| NVIDIA NeMo (`stt_ru_conformer_transducer_large`, `parakeet-tdt-0.6b-v3`, TitaNet) | CC-BY-4.0 требует атрибуции; отдельно проверить лицензии датасетов (Golos, Common Voice, RuLS, SOVA) и условия распространения весов. |
| 3D-Speaker / CAM++ / ERes2Net | Apache-2.0 на код; проверить, что веса на ModelScope распространяются под той же лицензией (в карточке ModelScope указать явно). |
| WeSpeaker | Apache-2.0 на код; проверить лицензию конкретных pretrained-весов (VoxCeleb и др.). |
| Whisper (веса OpenAI) | MIT; проверить, что используемая ggml-конвертация не меняет лицензию (обычно MIT). |
| Vosk models | Apache-2.0; но runtime Vosk/Kaldi вне разрешённого списка — использовать только sherpa-ONNX-конвертации. |
| Moonshine | MIT для большинства моделей, но legacy non-English — NC; русского STT нет, так что неприменимо. |

### Общие правила
- CC-BY-4.0 — разрешено, но требует видимой атрибуции в приложении/документации.
- Любую NC-модель нельзя включать в APK даже как «опциональную загрузку» без отдельного юридического решения.
- Для каждой модели фиксировать: точный URL, тег/ревизию, дату скачивания, хэш файла — иначе воспроизводимость и лицензионный аудит невозможны.

---

## Открытые вопросы (для следующего шага)
1. Есть ли у GigaAM-v3 официальный/поддерживаемый ONNX-экспорт с int8 и token timestamps? (сейчас — только сторонние `onnx-asr`-пространства).
2. Какой реальный DER даёт связка pyannote segmentation-3.0 + CAM++ на русских 2–3-спикерных записях? (нужен diar-engineer).
3. Нужен ли streaming ASR в v1, или достаточно offline + VAD? От этого зависит выбор T-one vs Zipformer.
4. Подтвердить у Silero письменно MIT для VAD (из-за противоречия бейджа и LICENSE).
