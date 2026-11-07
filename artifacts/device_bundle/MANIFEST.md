# device_bundle — Samsung SM-N9750 (Android 12, arm64-v8a)

Target: offline pipeline ASR (sherpa-onnx small-zipformer-ru int8) → diarization
(pyannote segmentation + 3dspeaker CAM++ embedding) → LLM post-processing
(MiniCPM5-2B + LoRA smoke, ORT GenAI int4).

## Contents

| Path | Size (bytes) | Notes |
|---|---:|---|
| `models/asr/tokens.txt` | 6388 | sherpa-onnx tokens |
| `models/asr/encoder.int8.onnx` | 25090964 | sha256 `75653125BF3621D86A2B140FFD84A1C3B7610FC8CD49F2B6D70CF5786BE0D085` |
| `models/asr/decoder.int8.onnx` | 1307950 | |
| `models/asr/joiner.int8.onnx` | 259572 | |
| `models/diar/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx` | 28281138 | sha256 `F682B514C05D947EE3FA91CD6EC6C5C7543479A128373FA29B1FAEDCCD21FD11` |
| `models/diar/pyannote-segmentation-3.0/model.int8.onnx` | see dir | segmentation model (if present) |
| `models/llm/model.onnx` | 300488 | sha256 `CFDA300EA8229311020FE55A7FF0DC3571D0FE0D071936A43EF14D9CA95963FE` |
| `models/llm/model.onnx.data` | 1640669184 | sha256 `AD4F25ECACD66710A6AF9DAFFA819F822842E0798405047A33FE9E7E32ECF6C3` |
| `models/llm/genai_config.json` | 1635 | sha256 `8071F3BCA5940D5B84B7B4A30608C113D8272F9BD3DA4E31666CAB62B705A2E5` |
| `models/llm/tokenizer.json` | 9894271 | MiniCPM5 tokenizer (see blocker below) |
| `models/llm/tokenizer_config.json` | 452 | |
| `models/llm/chat_template.jinja` | 5547 | ChatML, enable_thinking=false |
| `bin/sherpa/libonnxruntime.so` | 22249560 | sherpa-onnx v2.0.1 android arm64-v8a |
| `bin/sherpa/libsherpa-onnx-c-api.so` | 4303136 | |
| `bin/sherpa/libsherpa-onnx-cxx-api.so` | 272848 | |
| `bin/sherpa/libsherpa-onnx-jni.so` | 4609320 | |
| `bin/sherpa/sherpa-onnx-offline` | 23868168 | termux-static v2.0.1 CLI (needs libhashshim.so) |
| `bin/sherpa/sherpa-onnx-offline-speaker-diarization` | 22216648 | termux-static v2.0.1 CLI |
| `bin/sherpa/libhashshim.so` | 5304 | `std::__ndk1::__hash_memory` shim (NDK r27c) |
| `models/diar/pyannote-segmentation-3.0/model.fixed.onnx` | 6153666 | topo-sorted pyannote seg (MIT) |
| `models/diar/pyannote-segmentation-3.0/model.int8.fixed.onnx` | 1540506 | topo-sorted int8 |
| `audio/ruls_0001.wav` | 636204 | ASR smoke |
| `audio/diar_0002.wav` | 1072282 | diarization smoke |
| `audio/diar_0003.wav` | 1166160 | diarization smoke |
| `src/llm_cli.cpp` | 5193 | ORT GenAI CLI (not built in this run) |

Bundle total: ~1.62 GB.

## Versions / licenses

- sherpa-onnx **v2.0.1** — Apache-2.0 (k2-fsa/sherpa-onnx).
- ONNX Runtime GenAI **0.19.0** — MIT (microsoft/onnxruntime-genai).
- MiniCPM5-2B base + LoRA `smoke` merged (bf16) → ORT GenAI int4 (block 32, accuracy 4) — Apache-2.0 (per project).
- pyannote segmentation-3.0 — MIT; 3dspeaker CAM++ — Apache-2.0 (see upstream).
- Android NDK r27c (clang aarch64-linux-android24).

## Exact commands

Push:
```
powershell -ExecutionPolicy Bypass -File scripts\push_to_device.ps1
```

Run stages (wall-time printed):
```
powershell -ExecutionPolicy Bypass -File scripts\run_stage.ps1 -Stage asr  -Audio ruls_0001.wav
powershell -ExecutionPolicy Bypass -File scripts\run_stage.ps1 -Stage diar -Audio diar_0002.wav
powershell -ExecutionPolicy Bypass -File scripts\run_stage.ps1 -Stage llm
powershell -ExecutionPolicy Bypass -File scripts\run_stage.ps1 -Stage all
```

Manual ASR:
```
adb -s RF8N10DPSDL shell "cd /sdcard/pipeline && LD_LIBRARY_PATH=/sdcard/pipeline/bin/sherpa \
  ./bin/sherpa/sherpa-onnx-offline --tokens=models/asr/tokens.txt \
  --encoder=models/asr/encoder.int8.onnx --decoder=models/asr/decoder.int8.onnx \
  --joiner=models/asr/joiner.int8.onnx audio/ruls_0001.wav"
```

## Status (2026-11-07, profiler run)

1. **ASR — WORKS on device.** `sherpa-onnx-offline` (termux-static v2.0.1) runs
   with `LD_PRELOAD=libhashshim.so` (NDK r27c libc++ lacks
   `std::__ndk1::__hash_memory`). Binaries must live on an exec mount:
   `/data/local/tmp/pipeline/bin/sherpa`; models/audio stay on `/sdcard/pipeline`.
   Measured: ruls_0001 wall 8.3–8.8 s, decode RTF 0.038–0.040, VmHWM ~306 MB.
   ASR crashes (Reshape) on audio >~20 s — needs chunking (not in CLI).
2. **Diarization — WORKS on device.** pyannote seg `model.fixed.onnx` (topo-sorted;
   original fails ORT 1.22 "Nodes in a graph must be topologically sorted") +
   3dspeaker CAM++, `--clustering.cluster-threshold=0.7`. Measured RTF 0.31–0.40
   (decode), VmHWM 149–163 MB.
3. **LLM — NOT measured.** ORT GenAI 0.19.0 (latest) rejects MiniCPM5
   `tokenizer.json` (`\p{L}`/`\p{N}`); LiteRT-LM v0.19.1 has no Android prebuilt
   (macOS/iOS only) and no bazel toolchain is available. See
   `reports/profiler/DEVICE_REPORT.md` §6.

Full measurements: `reports/profiler/DEVICE_REPORT.md`; raw stdout:
`reports/profiler/device_runs/`.

## Addendum (2026-11-07, profiler round 2): LLM on-device + chunked ASR

### llm_cli (built)
- `bin/llm/llm_cli` — 67,976 bytes, NDK r27c `aarch64-linux-android24-clang++`, flags `-O2 -fPIE -pie -fno-emulated-tls -nostdlib++` (links libmat's libc++), `-lonnxruntime-genai -lmat -llog -ldl -lm`.
- Runtime libs (same dir, `LD_LIBRARY_PATH`): `libonnxruntime-genai.so` (6,255,112; ORT GenAI 0.19.0, MIT), `libmat.so` (28,039,168; ORT GenAI Android AAR, MIT), `libonnxruntime.so` (19,242,472; ONNX Runtime 1.23.0 Android AAR, MIT — dlopen'd by GenAI).
- `libc++_shared.so` (1,794,776; NDK r27c, Apache-2.0 with LLVM exceptions) is present but NOT required by the final `-nostdlib++` build.
- Build notes: bionic arm64 rejects underaligned PT_TLS from lld → `alignas(64) thread_local` pad + `-fno-emulated-tls`. Missing `libonnxruntime.so` was the original model-load blocker (GenAI dlopens it by name).

### QVikhr int4 (device copy)
`/sdcard/pipeline/models/llm/QVikhr-3-1.7B-Instruction-noreasoning-int4/` (1,108,468,904 bytes total):

| File | Bytes | SHA256 |
|---|---:|---|
| model.onnx | 290,594 | F873CEBE5159D4100DA0063653A8D3FB3AF563E5C5B83BCFF1EC1A42D6F06AC0 |
| model.onnx.data | 1,096,749,056 | 2F74EEB412E745F00D88BFFE108E4AEE365825933FCD817ECFD3215E86C3E265 |
| genai_config.json | 1,668 | 160168836C9C6B4D981A586AE0A95D8A29A9D29E2DF8631F7B3027E0F6478D42 |
| tokenizer.json | 11,422,650 | BE75606093DB2094D7CD20F3C2F385C212750648BD6EA4FB2BF507A6A4C55506 |
| chat_template.jinja | 4,214 | 3BD7D7E5B167E8B424759C8841AE488B2252167F7FDDD6F8097FBEA77C9B67C2 |
| tokenizer_config.json | 722 | 70DBFF0C676DB3313A55D21970D115B24A9BBC3F8BF7682C7D73CC948C7F7344 |

### Chunked ASR mode
`scripts/run_stage.ps1 -Stage asr -Audio <file> -Chunked [-ChunkSec 19]`: splits WAV on PC (`.venv` + soundfile) into ≤19 s chunks, pushes to `/sdcard/pipeline/audio/chunks/`, runs ASR per chunk via device `bench.sh`, concatenates text, sums wall/decode time. Measured diar_0002/0003: wall RTF 0.47–0.52, decode RTF 0.038–0.039, VmHWM ~298 MB.
