# GigaAM v3 RNN-T (sherpa-onnx ONNX) — MANIFEST

- **Model**: GigaAM v3 RNN-T, NeMo `EncDecRNNTBPEModel` exported to ONNX (sherpa-onnx `model_type=nemo`).
- **Source repo**: https://github.com/DaniilYemelyanenka/rashifruy
- **Path in repo**: `src/mobile/android/app/src/main/assets/gigaam/`
- **Revision**: `baa40b5e893b906762ae420a882ef50db031661d` (main, 2026-11-07T15:10:29Z)
- **Upstream model**: https://github.com/salute-developers/GigaAM (v3 line)
- **License**: **MIT** — upstream `LICENSE` (Copyright (c) 2024 GigaChat Team), https://github.com/salute-developers/GigaAM/blob/main/LICENSE
- **v3 confirmation**: encoder ONNX metadata contains `comment=v3`, `version=3`, `model_type=EncDecRNNTBPEModel`, `model_author=https://github.com/salute-developers/GigaAM`, `license=https://github.com/salute-developers/GigaAM/blob/main/LICENSE`, `is_giga_am=1`, `vocab_size=1024`, `subsampling_factor=4`, `pred_rnn_layers=1`, `pred_hidden=320`.

## Files

| file | bytes | SHA-256 | URL |
|---|---:|---|---|
| encoder.int8.onnx | 224,570,820 | `369f35a71bf288d3b8e0391fabd8dba5f2314088d440bca474056b7b4b6e66bf` | https://media.githubusercontent.com/media/DaniilYemelyanenka/rashifruy/main/src/mobile/android/app/src/main/assets/gigaam/encoder.int8.onnx |
| decoder.onnx | 4,600,132 | `38fc7475443ea2a26f63211ca350f73ac50fff824ab7a3876ee2bd610c53bbc4` | https://media.githubusercontent.com/media/DaniilYemelyanenka/rashifruy/main/src/mobile/android/app/src/main/assets/gigaam/decoder.onnx |
| joiner.onnx | 2,712,896 | `602ff7017a93311aad34df1437c8d7f49911353c13d6eae7a6ee7b041339465c` | https://media.githubusercontent.com/media/DaniilYemelyanenka/rashifruy/main/src/mobile/android/app/src/main/assets/gigaam/joiner.onnx |
| tokens.txt | 13,354 | `39abae20e692998290c574e606f11a9edef2902a1995463fcff63d1490cf22b7` | https://raw.githubusercontent.com/DaniilYemelyanenka/rashifruy/main/src/mobile/android/app/src/main/assets/gigaam/tokens.txt |

- **Total size**: 231,897,202 bytes = **221.15 MiB**.
- ONNX files are stored via Git LFS in the source repo (media.githubusercontent.com); `tokens.txt` is a regular file (raw.githubusercontent.com).
- `tokens.txt`: 1025 lines, blank `<blk>` id 1024 (last), matches `vocab_size+1` expected by sherpa-onnx NeMo decoder.
