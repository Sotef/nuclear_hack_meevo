# models/diar — MANIFEST

Дата скачивания: 2026-11-06. Источник: GitHub releases `k2-fsa/sherpa-onnx`.
Все файлы скачаны в `D:\Downloads_D\nuclear_hack\models\diar`.

| Файл | Роль | Версия / тег | URL | Лицензия | Размер | SHA256 |
|---|---|---|---|---|---|---|
| `sherpa-onnx-pyannote-segmentation-3-0/model.onnx` | сегментация речи/спикеров (pyannote segmentation-3.0, fp32) | `speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0` | https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 | MIT (LICENSE в архиве, © 2022 CNRS) | 5.99 MB (архив 6.6 MB) | архив: `24615EE884C897D9D2BA09BB4D30DA6BB1B15E685065962DB5B02E76E4996488` |
| `sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx` | та же сегментация, int8 (не использовалась в baseline) | тот же архив | тот же | MIT | 1.54 MB | — |
| `3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx` | эмбеддинги спикера (3D-Speaker CAM++, zh-cn, 16 kHz) | `speaker-recongition-models` (так в URL релиза) | https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx | Apache-2.0 (3D-Speaker; см. CANDIDATES.md) | 27.0 MB | `F682B514C05D947EE3FA91CD6EC6C5C7543479A128373FA29B1FAEDCCD21FD11` |
| `nemo_en_titanet_small.onnx` | эмбеддинги спикера (NVIDIA NeMo TitaNet-small, en) — только диагностический прогон C | `speaker-recongition-models` | https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx | CC-BY-4.0 (NVIDIA NeMo; требуется атрибуция) | 38.4 MB | `AD4A1802485D8B34C722D2A9D04249662F2ECE5D28A7A039063CA22F515A789E` |

## Примечания

- `model.onnx` (fp32) использован во всех прогонах A/B/C/D; int8-вариант скачан, но не оценивался (квантизация — зона optimizer).
- CAM++ — основная модель эмбеддингов (конфигурации A и B). TitaNet-small — только дополнительная проверка C, в A/B не входит.
- Лицензии проверены по LICENSE-файлу архива (pyannote) и карточкам источников из `reports/model_scout/CANDIDATES.md` (CAM++ Apache-2.0, TitaNet CC-BY-4.0).
- Версия рантайма: `sherpa-onnx` Python 2.0.1 (`sherpa-onnx-core` 2.0.1).
