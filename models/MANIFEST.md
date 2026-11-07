# Манифест скачанных моделей

Дата скачивания: 2026-11-06. Автор: asr-engineer.
Источник архивов: GitHub releases `k2-fsa/sherpa-onnx`, тег `asr-models`.
Все модели — offline transducer (encoder/decoder/joiner) для sherpa-onnx.

## A. sherpa-onnx-zipformer-ru-2024-09-18

| Поле | Значение |
|---|---|
| Имя | `sherpa-onnx-zipformer-ru-2024-09-18` |
| Версия / дата релиза | 2024-09-18 |
| Донор | `alphacep/vosk-model-ru` (Vosk/Kaldi → ONNX) |
| URL архива | https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ru-2024-09-18.tar.bz2 |
| Документация | https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html |
| Лицензия | **Apache-2.0** (Vosk model-ru, https://alphacephei.com/vosk/models) — разрешено |
| Размер архива | 297 360 353 байт (283.6 MiB) |
| Размер на диске (распаковано) | 332 334 196 байт (316.9 MiB), включая fp32 и int8 |
| int8-комплект (использован) | encoder.int8 68 187 180 + decoder.int8 1 307 950 + joiner.int8 259 572 + tokens 6 388 + bpe 246 184 = **70 007 274 байт (66.8 MiB)** |

SHA-256 (int8-комплект):

| Файл | Байт | SHA-256 |
|---|---:|---|
| encoder.int8.onnx | 68 187 180 | `69A347DDF7E5D39452D17F059DFEDCD8C9471CA1953C320CB04CA19626149FDB` |
| decoder.int8.onnx | 1 307 950 | `5B83B80D557538517F010F64E1961BA57FD9DE2B93BF73B77B8D27F0AC28160F` |
| joiner.int8.onnx | 259 572 | `A6F550B56375326EE4A67DCAC2CAC027035E2DC137C08FFF3557F4FFDC5CBC3E` |
| tokens.txt | 6 388 | `93BBBC0BAE6B78C0BBB743D4AA9FDED3BB5FF3AAC5F0200E3A769A5A05E0FDF6` |
| bpe.model | 246 184 | `C7A756AEB3550417D6B2ED3EFDE9A7AA3EEA54787D4EAC011E9CCE6090C9C64A` |

Примечание: в рантайме использован `decoder.onnx` (fp32, 2 093 079 байт), как в официальной
команде sherpa-onnx; `decoder.int8.onnx` в комплекте есть, но для декодера рекомендуется fp32.

## B. sherpa-onnx-small-zipformer-ru-2024-09-18

| Поле | Значение |
|---|---|
| Имя | `sherpa-onnx-small-zipformer-ru-2024-09-18` |
| Версия / дата релиза | 2024-09-18 |
| Донор | `alphacep/vosk-model-small-ru` (Vosk/Kaldi → ONNX) |
| URL архива | https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-small-zipformer-ru-2024-09-18.tar.bz2 |
| Документация | https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html |
| Лицензия | **Apache-2.0** (Vosk model-small-ru, https://alphacephei.com/vosk/models) — разрешено |
| Размер архива | 109 906 654 байт (104.8 MiB) |
| Размер на диске (распаковано) | 120 281 260 байт (114.7 MiB), включая fp32 и int8 |
| int8-комплект (использован) | encoder.int8 25 090 964 + decoder.int8 1 307 950 + joiner.int8 259 572 + tokens 6 388 + bpe 246 184 = **26 911 058 байт (25.7 MiB)** |

SHA-256 (int8-комплект):

| Файл | Байт | SHA-256 |
|---|---:|---|
| encoder.int8.onnx | 25 090 964 | `75653125BF3621D86A2B140FFD84A1C3B7610FC8CD49F2B6D70CF5786BE0D085` |
| decoder.int8.onnx | 1 307 950 | `435752764FB93C3E3E8D2537B9D008700114C7E126461E7808DE30EF4AED4696` |
| joiner.int8.onnx | 259 572 | `A93E0AFC2C82E94D366157B4A8B5F4CBA6973DDEA8F36AC94010B430BE7823F1` |
| tokens.txt | 6 388 | `93BBBC0BAE6B78C0BBB743D4AA9FDED3BB5FF3AAC5F0200E3A769A5A05E0FDF6` |
| bpe.model | 246 184 | `C7A756AEB3550417D6B2ED3EFDE9A7AA3EEA54787D4EAC011E9CCE6090C9C64A` |

Примечание: в рантайме использован `decoder.onnx` (fp32, 2 093 080 байт), как в официальной
команде sherpa-onnx.

## Общее

- Обе модели — ru-only, offline (не streaming), token-level timestamps.
- Архивы `.tar.bz2` оставлены в `models/` для воспроизводимости; распакованные каталоги рядом.
- Лицензия проверена по карточке Vosk models (Apache-2.0) и по документации sherpa-onnx.
