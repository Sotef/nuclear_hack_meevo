# MiniCPM5-2B: LoRA-дообучение на T-Wix и postproc-метрики

Дата: 2026-11-07. Автор: asr-engineer.
Модель: `openbmb/MiniCPM5-2B`, ревизия `f97400052a43d642bbc6e9975e2397e3ae6a6b52`
(Apache-2.0 по метаданным HF; файла LICENSE в репозитории нет).
Данные: `data/twix/train_smoke.jsonl` (10 000 строк, seed=42) и
`data/twix/train_full.jsonl` (499 598 строк), формат `messages` (SFT-микс, не пары пунктуации).
Оценка: `tools/eval_postproc` 1.0.0 на `data/postproc/punct_eval.jsonl` (300 пар) и
`data/postproc/correction_eval.jsonl` (13 пар). Промпт: `prompts/postproc_v1.md` (не изменялся).
Все прогоны — greedy (temperature=0), torch/transformers, CUDA bf16, RTX 3080 10 ГБ.

## 1. Диск и размещение

| Что | Значение |
|---|---|
| Свободно до старта | C: 5.82 ГБ, D: 4.01 ГБ |
| Удалено | `data/twix/raw/` (8 parquet, 1.58 ГБ) — **только после проверки**, что `train_smoke.jsonl`, `train_full.jsonl`, `stats.json` уже выгружены. Parquet можно повторно скачать из `t-tech/T-Wix` @ `0059b0967915624069b5b4dc224ebe7ef7c041a6`. |
| HF_HOME | остался `D:\hf_cache` (5.33 ГБ сторонних моделей, не трогался) |
| Веса bf16 | скачаны в `models/llm_minicpm5/base` (4.70 ГиБ), затем **перенесены** в `C:\temp\opencode\minicpm5_base`, чтобы освободить D: под чекпоинты |
| Свободно после | C: 1.58 ГБ, D: 5.73 ГБ |
| Правка локальной копии | в `tokenizer_config.json` `eos_token` заменён на `<|im_end|>` (id 130073), чтобы `run_llm` останавливался на chat-end токене; `generation_config.json` уже содержит `eos_token_id: [1, 130073]` |

## 2. Конфиг обучения (smoke)

| Параметр | Значение |
|---|---|
| Метод | LoRA bf16 (без QLoRA/bitsandbytes) |
| LoRA | r=16, alpha=32, dropout=0.05, bias=none, task=CAUSAL_LM |
| target_modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| Обучаемых параметров | 25 116 672 (0.988% от 2 541 873 152) |
| lr | 2e-4, cosine, warmup 75 шагов |
| max_seq_len | 512 |
| batch / grad_accum | 1 / 4 (effective batch 4) |
| epochs | 1 |
| steps | 2 500 |
| seed | 42 (python/numpy/torch/cuda, cudnn deterministic) |
| optimizer | adamw_torch, grad checkpointing, bf16 |
| Данные | `train_smoke.jsonl`, 10 000 записей, 0 пропусков |
| Время | **6 518 с = 1.811 ч** |
| Пик CUDA | **6.350 ГБ** |
| Train loss | **1.1028** (первый шаг 1.165, последний 1.089, mean 1.103, min 0.878, max 1.385, NaN нет) |
| Лог | `logs/train_minicpm5_smoke.log` |
| Адаптер | `models/llm_minicpm5/smoke/` (`adapter_model.safetensors` 100.5 МБ, `train_summary.json`) |

Loss по всему прогону колеблется в диапазоне ~0.88–1.38 и снижается лишь умеренно
(1.165 → 1.089). Это ожидаемо: T-Wix — общий SFT-микс, а не пары пунктуации; 1 эпоха
на 10k примеров и seq 512. Признаков деградации (рост/NaN) нет, поэтому full-прогон
был разрешён.

## 3. Как считались метрики и важная оговорка про prompt

`run_llm` 1.1.0 собирает ChatML вручную и заканчивает промпт на
`<|im_start|>assistant\n`, **не вставляя** пустой блок ` thinking\n\n</think>\n\n`,
который MiniCPM5 использует при `enable_thinking=false`. На 2 примерах `punct_eval`
сырой `run_llm` дал **0 символов** (137/163 токена английского рассуждения, обрезанного
лимитом). С родным chat-шаблоном модели (`add_generation_prompt=True`,
`enable_thinking=False`) тот же system/user-промпт даёт прямой ответ.

Поэтому для MiniCPM5 использован тонкий wrapper `D:\temp\opencode\run_llm_minicpm.py`:
он подменяет **только** `build_prompt` (assistant-префикс берётся из chat-шаблона модели),
а цикл генерации, greedy, eos, формат JSONL и метрики — код `run_llm`. Файл
`prompts/postproc_v1.md` не изменялся. Для smoke-адаптера wrapper дополнительно
загружает PEFT-адаптер поверх base (без merge). Это отклонение от буквальной команды
`run_llm --model ...`, необходимое для корректного замера MiniCPM5.

## 4. Итоговая сравнительная таблица

Punct-колонки — `punctuate` на `punct_eval` (300 пар); WER-колонки — `correct` на
`correction_eval` (13 пар). Для QVikhr и MiniCPM5 это два разных прогона, как и в
`COMPARE.md`; строки не складываются в одну «общую» оценку.

| Строка | Punct P / R / F1 (micro) | Boundary F1 | WER before → after | ΔWER | Improved / worsened / unchanged |
|---|---|---:|---|---:|---|
| **no-LLM** (сырой ASR) | 1.000000 / 0.000000 / **0.000000** | 0.000000 | 0.146881 → 0.146881 | 0.000000 | 0 / 0 / 13 |
| **QVikhr int4** (punctuate + correct+punct) | 0.817875 / 0.703919 / **0.756630** | 0.848580 | 0.146881 → 0.142857 | **−0.004024** | 1 / 3 / 9 |
| **MiniCPM5-2B base** (zero-shot) | 0.917431 / 0.290276 / **0.441014** | 0.770642 | 0.146881 → 0.146881 | 0.000000 | 1 / 1 / 11 |
| **MiniCPM5-2B smoke-tuned** | 0.827922 / 0.740203 / **0.781609** | 0.870662 | 0.146881 → 0.156942 | **+0.010060** | 1 / 3 / 9 |
| **MiniCPM5-2B full-tuned** | не измерено | не измерено | не измерено | не измерено | не измерено |

Дополнительно (macro):

| Строка | Punct macro P / R / F1 | Macro boundary F1 |
|---|---|---:|
| QVikhr punctuate | 0.901426 / 0.726444 / 0.753214 | — |
| MiniCPM5 base | 0.941667 / 0.304278 / 0.409667 | 0.682444 |
| MiniCPM5 smoke | 0.868256 / 0.769444 / 0.793591 | 0.878857 |

**Выводы по качеству.**
- Zero-shot MiniCPM5 ставит мало знаков: precision 0.917, recall 0.290, F1 0.441.
  Это ниже QVikhr (F1 0.757).
- Smoke-LoRA резко поднимает recall (0.290 → 0.740) и F1 до **0.7816**, boundary F1 до
  **0.8707** — выше QVikhr (0.7566 / 0.8486) на этом наборе. Precision при этом падает
  (0.917 → 0.828), т.е. модель начинает ставить больше знаков, часть — лишние.
- В режиме `correct` smoke-адаптер **ухудшает** WER на correction-13
  (+0.0101, worsened 3/13), тогда как base не меняет WER (0.0000), а QVikhr немного
  улучшает (−0.0040). Для коррекции smoke-адаптер в текущем виде не рекомендуется.
- Full-tuned метрики не измерены: full-прогон не завершён (см. §6).

## 5. Скорость (GPU, greedy, RTX 3080)

| Прогон | Элементов | Сек/элемент | Decode, ток/с | Пик CUDA, ГБ |
|---|---:|---:|---:|---:|
| MiniCPM5 base, punctuate | 300 | 0.831 | 19.63 | 5.085 |
| MiniCPM5 smoke, punctuate | 300 | 1.940 | 9.62 | 5.234 |
| MiniCPM5 base, correct | 13 | 4.640 | 18.23 | 5.138 |
| MiniCPM5 smoke, correct | 13 | 9.244 | 9.13 | 5.239 |

Smoke-адаптер в текущем виде (PEFT без merge) примерно в 2 раза медленнее base на
генерации; это цена adapter-слоёв и большего числа сгенерированных токенов. Цифры —
GPU-стенд, не Android.

## 6. Full-прогон: статус

Full запущен в фоне на `train_full.jsonl` (499 598 строк) с тем же конфигом
(seq 512, batch 1, grad_accum 4, 1 эпоха, save_steps=0 — промежуточные чекпоинты
отключены из-за диска).

| Параметр | Значение |
|---|---|
| Лог | `logs/train_minicpm5_full.log` |
| PID-файл | `logs/train_minicpm5_full.pid` |
| Шагов всего | 124 900 |
| Дошло до | **31 шаг** (loss 1.071 / 1.459, grad_norm ~1.0–1.4) |
| Скорость | ~2.57 с/шаг |
| ETA | **~89 ч 06 мин** |
| Статус | **остановлен** (Stop-Process) из-за бюджета и роста pagefile на D: (свободно падало до 1.27 ГБ); адаптер full не сохранён |
| Full-tuned eval | не измерено |

Команда повторного запуска (из `D:\Downloads_D\nuclear_hack`):

```
.venv\Scripts\python.exe tools\train_llm\train_llm.py --model C:\temp\opencode\minicpm5_base --data data\twix\train_full.jsonl --out models\llm_minicpm5\full --epochs 1 --lr 2e-4 --lora-r 16 --lora-alpha 32 --max-seq-len 512 --batch 1 --grad-accum 4 --save-steps 0 --log logs\train_minicpm5_full.log --force
```

Мониторинг:

```
Get-Content logs\train_minicpm5_full.log -Tail 20
```

Остановка:

```
$procId = [int](Get-Content logs\train_minicpm5_full.pid); Stop-Process -Id $procId -Force
```

## 7. Ограничения

- **T-Wix — не про пунктуацию.** Это общий SFT-микс (general/reasoning/long_context/
  english_corpus), пар «сырой ASR → исправленный текст» и «без пунктуации → с пунктуацией»
  в нём нет. Прирост punct F1 — следствие русского/инструктивного SFT, а не прямой
  супервизии по пунктуации.
- **RU у MiniCPM5 не заявлен.** Карточка: `language: en, zh`; RU-бенчмарков нет.
  Токенизация кириллицы отдельно не измерялась.
- **Assistant-only loss** реализован через маркеры chat-шаблона (`<|im_start|>assistant`
  … `<|im_end|>`); при отсутствии assistant-спана — fallback на полный LM-loss.
  Официальный `{% generation %}`-патч не применялся.
- **seq 512** обрезает большинство примеров: в первых 200 строках smoke медиана полной
  длины 512 токенов, медиана assistant-токенов 393.5. Длинные ответы теряются.
- **1 эпоха / 10k строк**: loss снизился умеренно (1.165 → 1.089), без NaN.
- **correction-13** — всего 13 пар; шаг improved/worsened = 1/13, статистика грубая.
- **Base zero-shot** имеет высокий precision и низкий recall: модель ставит мало знаков.
- **Smoke correct** ухудшает WER (+0.0101) на 13 парах; для продукта в режиме коррекции
  не рекомендуется без дополнительного обучения на парах ASR→ref.
- **Full-tuned** не обучен и не измерен; ETA полного прогона ~89 ч на RTX 3080.
- **Скорость** — GPU-стенд с занятой desktop-памятью (~3.1 ГБ), не Android.
- **Лицензия**: Apache-2.0 подтверждается метаданными HF, файла LICENSE в репозитории нет;
  лицензии данных T-Wix (ODC-BY-1.0) и third-party outputs требуют отдельного аудита.

## 8. Артефакты

- Инструмент: `tools/train_llm/train_llm.py` v1.0.0 (запись в `tools/TOOLS.md`)
- Логи: `logs/train_minicpm5_smoke.log`, `logs/train_minicpm5_full.log`,
  `logs/train_probe.log`, `logs/download_minicpm5.log`
- Адаптер smoke: `models/llm_minicpm5/smoke/` (`adapter_model.safetensors`,
  `adapter_config.json`, `train_summary.json`)
- Full: `models/llm_minicpm5/full/` — адаптер не сохранён (прогон остановлен)
- Base-веса: `C:\temp\opencode\minicpm5_base` (перенесены с D:)
- Гипотезы/метрики: `reports/postproc_eval/minicpm5_{base,smoke}_{punct300,correct13}.jsonl`,
  `..._metrics.json`, `..._eval.json`, `..._eval.txt`
- Wrapper: `D:\temp\opencode\run_llm_minicpm.py` (scratch, не инструмент проекта)
- Журнал: `logs/journal.md`
