# Сравнение постпроцессора: без LLM vs QVikhr-3-1.7B

Дата: 2026-11-07. Инструмент: `tools/eval_postproc` 1.0.0 (Python 3.12.10).
Все числа — один инструмент, одни и те же референсы:
`data/postproc/punct_eval.jsonl` (300 пар) и `data/postproc/correction_eval.jsonl` (13 пар).
LLM-прогоны — greedy (temperature=0), torch/transformers, CUDA bf16 (RTX 3080).

## Сводная таблица

| Строка | Punct P / R / F1 (micro) | Boundary F1 | WER before → after | ΔWER | Improved / worsened / unchanged | False-alarm (clean-50) | Покрытие ASR-ошибок |
|---|---|---:|---|---:|---|---:|---:|
| **no-LLM** (сырой ASR, без пунктуации) | 1.000000 / 0.000000 / **0.000000** | 0.000000 | 0.146881 → 0.146881 | 0.000000 | 0 / 0 / 13 | 0.000000 (0/50) | 0.000000 (0/119) |
| **QVikhr punctuate** (punct300) | 0.817875 / 0.703919 / **0.756630** | 0.848580 | — (не измерено) | — | — | 0.000000 (0/50)* | 0.000000 (0/119)* |
| **QVikhr correct+punct** (correct13_punct) | — (не измерено) | — | 0.146881 → 0.142857 | **−0.004024** | 1 / 3 / 9 | — (не измерено) | 0.000000 (0/119)* |
| **QVikhr full** (full63) | 0.853659 / 0.299145 / **0.443038**† | 0.202899† | 0.146881 → 0.179074 | **+0.032193** | 0 / 4 / 9 | 0.160000 (8/50) | 0.025210 (3/119) |

\* Ноль тривиален: режимы `punctuate`/`correct` не выдают `doubtful`-флаги и структурированные `edits`, поэтому false-alarm и coverage равны 0 не потому, что флаги идеальны, а потому что флагов нет.
† Пунктуация `full` измерена только на clean-50 (подмножество `punct_eval`), а не на всех 300; WER — на correction-13. Строка не полностью сопоставима с punct300 по объёму.

## Источники чисел

| Строка | Файлы-источники |
|---|---|
| no-LLM | `reports/postproc_eval/no_llm_punct300.json/.txt`, `no_llm_correct13.json/.txt` (этот прогон) |
| QVikhr punctuate | `reports/postproc_eval/llm_torch_punct300_eval.json/.txt`; `reports/asr_engineer/POSTPROC_REPORT.md` §2.1 |
| QVikhr correct+punct | `reports/postproc_eval/llm_torch_correct13_punct_eval.json/.txt` (пересчёт `eval_postproc` по `llm_torch_correct13_punct.jsonl`); метрики скорости — `llm_torch_correct13_punct_metrics.json` |
| QVikhr full | `reports/postproc_eval/llm_torch_full63_eval.json/.txt`; `reports/asr_engineer/POSTPROC_REPORT.md` §2.3 |

## Примечания

1. **Разрыв no-LLM → QVikhr punctuate** — это цена/выигрыш восстановления пунктуации: сырой ASR не ставит знаков вообще (recall 0, F1 0), QVikhr даёт F1 0.757 и boundary F1 0.849. Разрыв по WER для `punctuate` не измерен: прогон был только на punct-наборе, где нет ASR-эталонов с WER-разметкой.
2. **Коррекция**: `correct+punct` немного улучшает WER (−0.004), но улучшает 1 пару и ухудшает 3 из 13 — на 13 парах это статистически грубо. `full` заметно ухудшает WER (+0.032) и даёт 16 % false-alarm: модель перефразирует/галлюцинирует. Для продукта разумны `punctuate` + консервативный `correct`, `full` — экспериментальный режим.
3. **Статус чисел**: все метрики качества выше — **замеры на GPU** (torch bf16, RTX 3080), не на телефоне. Оценки скорости для Android (сек/мин, RTF, RAM) в `POSTPROC_REPORT.md` §4 — это **аналитические оценки**, а не замеры; не смешивать с измеренными метриками качества.
4. **Ограничения наборов**: punct_eval — письменные предложения Tatoeba (не спонтанная речь, не телефонный канал); correction_eval — 13 записей / 497 слов, ошибки ASR синтетические (baseline-гипотезы), пунктуации в эталонах нет. Поэтому punct- и WER-колонки измеряются на разных наборах и не складываются в одну «общую» оценку.
5. **Что именно такое строка `correct+punct`**: это существующий прогон `llm_torch_correct13_punct.jsonl` — режим `punctuate`, применённый к `correction_eval` (13 ASR-транскриптов). Это **не** последовательный пайплайн «сначала correct, потом punctuate». Пунктуация на этом наборе не измеряется, потому что в эталонах `correction_eval` нет знаков; WER после — 0.142857. Отдельного последовательного correct→punctuate прогона в отчётах нет.

## Дополнение: MiniCPM5-2B (2026-11-07)

Источник: `reports/asr_engineer/MINICPM_FT_REPORT.md`. Тот же `tools/eval_postproc` 1.0.0,
те же наборы (`punct_eval` 300, `correction_eval` 13) и тот же промпт `postproc_v1`;
greedy, torch bf16, RTX 3080. Для MiniCPM5 assistant-префикс берётся из родного
chat-шаблона модели (`add_generation_prompt=True`, `enable_thinking=False`), потому что
ручной ChatML в `run_llm` без пустого ` thinking`-блока уводит base-модель в рассуждение
и даёт пустой ответ (подробности и оговорки — в отчёте).

| Строка | Punct P / R / F1 (micro) | Boundary F1 | WER before → after | ΔWER | Improved / worsened / unchanged |
|---|---|---:|---|---:|---|
| **MiniCPM5-2B base** (zero-shot) | 0.917431 / 0.290276 / **0.441014** | 0.770642 | 0.146881 → 0.146881 | 0.000000 | 1 / 1 / 11 |
| **MiniCPM5-2B smoke-tuned** (LoRA r16, 10k, 1 эпоха) | 0.827922 / 0.740203 / **0.781609** | 0.870662 | 0.146881 → 0.156942 | **+0.010060** | 1 / 3 / 9 |
| **MiniCPM5-2B full-tuned** | не измерено | не измерено | не измерено | не измерено | не измерено |

Примечания:
- Punct-колонки — `punctuate` на `punct_eval` (300); WER-колонки — `correct` на
  `correction_eval` (13). Это два разных прогона, как и для QVikhr выше.
- Smoke-LoRA поднимает punct F1 до 0.7816 / boundary F1 0.8707 (выше QVikhr 0.7566 / 0.8486),
  но в режиме `correct` ухудшает WER (+0.0101) на 13 парах.
- Full-прогон запускался, но не завершён: 31/124900 шагов, ETA ~89 ч, остановлен из-за
  бюджета/диска; full-адаптер и его метрики отсутствуют.
- Файлы: `reports/postproc_eval/minicpm5_{base,smoke}_{punct300,correct13}_eval.json/.txt`,
  гипотезы `...jsonl`, метрики скорости `..._metrics.json`.
