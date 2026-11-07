# Ресерч: модели, замеры, лицензии

Полные отчёты экспериментов — `artifacts/test-results/reports/`, сводка
метрик — `artifacts/test-results/results.csv`. Все сравнения — на одном наборе
(`artifacts/data/test`, 13 записей, 497 слов, RuLS CC-BY-4.0 + синтетика MIT)
одним eval-инструментом (`src/tools/eval_asr`, `eval_postproc`, `eval_diar`).

## ASR (русский)

| Кандидат | WER micro | Размер | Лицензия | Статус |
|---|---:|---:|---|---|
| **GigaAM v3 RNN-T int8** | **0.121** | 221 MiB | MIT | ✅ принято (device) |
| small-zipformer-ru int8 | 0.147 | 25.7 MiB | Apache-2.0 | альтернатива (компактный) |
| zipformer-ru int8 (большой) | 0.175 | 67 MiB | Apache-2.0 | отклонён |
| Vosk small-ru | ~0.23 (опубл.) | 45 MiB | Apache-2.0 | не рассматривался (Kaldi-рантайм) |
| GigaAM v2 (sherpa) | — | 231 MiB | **NC** | ❌ запрещена лицензией |

Источник GigaAM v3 — `DaniilYemelyanenka/rashifruy` (export для sherpa), лицензия
сверена с upstream `salute-developers/GigaAM` (MIT). Замер RTF на устройстве:
decode 0.038 (zipformer) / 0.088 CPU и 0.082 GPU-encode (GigaAM).

## Пунктуация и регистр

| Кандидат | Punct F1 | Размер | Лицензия | Статус |
|---|---:|---:|---|---|
| **RUPunct-small int8 (ONNX)** | **0.838** | 28 MiB | MIT | ✅ принято |
| LLM QVikhr int4 | 0.757 | 1.1 GiB | Apache-2.0 | вытеснен (медленный) |
| Horizon-Labs truecase-small | 0.913 (по карточке) | 268 MiB | Apache-2.0 | запасной |
| CT-Transformer (sherpa) | 0.0 на RU | 72 MiB | Apache-2.0 | ❌ китайская, склеивает слова |

Формат RUPunct для порта в Kotlin: WordPiece (`do_lower_case=false`,
`add_prefix_space=true`), выход `logits [1,T,33]` — 3 режима регистра ×
11 классов пунктуации; агрегация «first subtoken»; boundary F1 0.948.

## Диаризация

- pyannote segmentation-3.0 (MIT) + 3D-Speaker CAM++ (Apache-2.0) +
  FastClustering (sherpa-onnx) — принято.
- DER 0.48 на синтетическом multi-speaker наборе (3 записи) — цифра завышена
  miss-ом из-за разметки (референс покрывает паузы внутри TTS-реплик);
  FA 0/10 на односпикеровых записях. Реальных разговорных записей с RTTM-разметкой
  в наборе нет («не измерено» честно зафиксировано).
- Отклонено: reverb-diarization (NC), pyannote diarization-3.1 (PyTorch-only,
  не для Android), початковая диаризация (нестабильные ID спикеров).

## LLM (конспект / главы / склейка фраз)

| Кандидат | RU-качество | Формат | Лицензия | Статус |
|---|---|---|---|---|
| **QVikhr-3-1.7B-Instruction int4** | punct F1 0.757 | ORT GenAI | Apache-2.0 | ✅ принято |
| MiniCPM5-2B + LoRA T-Wix (smoke) | punct F1 **0.782** | ORT GenAI (токенизатор падает) | Apache-2.0 | исследовательский |
| Qwen2.5-1.5B-Instruct | RU avg 49.8 (опубл.) | ORT GenAI | Apache-2.0 | запасной |
| Gamayun 1.5B | MERA 38.1 (статья) | — | **нет весов/лицензии** | ❌ |

На устройстве QVikhr: 15–16 ток/с, пик RAM 1.4–1.5 ГБ. Обучение LoRA
(T-Wix, smoke 10k): RTX 3080, 2500 шагов, 1.8 ч, loss 1.103; full-прогон
(499k строк, ETA ~89 ч) не запускался — экономически не оправдан.

## VAD

Silero VAD (MIT) — кандидат в пайплайн, пока не требуется: ASR и так
детерминированно обрабатывает тишину. Лицензионная оговорка: в README Silero
есть бейдж CC BY-NC при LICENSE=MIT — зафиксировано к юрпроверке.

## Нагрузочные замеры (SM-N9750)

| Показатель | Значение |
|---|---|
| Полный пайплайн, 7 мин аудио (без LLM) | RTF 0.44 (ASR 41.6 с, диар 153.7 с, пунктуация 0.2 с) |
| Полный пайплайн с LLM-анализом | RTF ≈ 1.06 (LLM-анализ ~2.5 мин) |
| Пик RAM | ASR 306 МБ, диаризация 163 МБ, LLM 1.4–1.5 ГБ |
| LLM | 15–16 ток/с, 6.9–19.9 с/мин аудио (лучше оценки 2–5×) |
| Загрузка моделей | GigaAM 2 с, полная инициализация ~50 с |

Источники: `artifacts/test-results/reports/profiler/DEVICE_REPORT.md`,
`asr_engineer/GIGAAM_VS_ZIPFORMER.md`, `RUPUNCT_REPORT.md`, `POSTPROC_REPORT.md`.

## Дизайн-исследование

Три варианта интерфейса (A Recorder / B Dot-Matrix / C Aurora Glass) с пулами
анимаций (ripple, пульс записи, эквалайзер, успех с галочкой) —
`artifacts/demo/design-3-variants.html`, реализованы в приложении как скины.
