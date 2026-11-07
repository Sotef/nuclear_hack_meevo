# run_stage.ps1 - run one pipeline stage on the device and print wall time.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\run_stage.ps1 -Stage asr|diar|llm|all [-Audio ruls_0001.wav] [-Chunked] [-ChunkSec 19]
# Binaries live in /data/local/tmp/pipeline/bin/sherpa (executable); models/audio on /sdcard/pipeline.
param(
  [ValidateSet('asr','diar','llm','all')][string]$Stage = 'all',
  [string]$Audio = 'ruls_0001.wav',
  [switch]$Chunked,
  [double]$ChunkSec = 19.0
)
$ErrorActionPreference = 'Stop'
$adb    = 'D:\temp\opencode\platform-tools\adb.exe'
$serial = 'RF8N10DPSDL'
$bin    = '/data/local/tmp/pipeline/bin/sherpa'
$dest   = '/sdcard/pipeline'
$ldenv  = "LD_LIBRARY_PATH=$bin LD_PRELOAD=$bin/libhashshim.so"
$py     = 'D:\Downloads_D\nuclear_hack\.venv\Scripts\python.exe'
$sw = [System.Diagnostics.Stopwatch]::StartNew()

function Run-Asr {
  Write-Output "--- ASR (sherpa-onnx offline zipformer int8) ---"
  & $adb -s $serial shell "cd /data/local/tmp/pipeline && $ldenv ./bin/sherpa/sherpa-onnx-offline --tokens=$dest/models/asr/tokens.txt --encoder=$dest/models/asr/encoder.int8.onnx --decoder=$dest/models/asr/decoder.int8.onnx --joiner=$dest/models/asr/joiner.int8.onnx $dest/audio/$Audio"
}

# Chunked ASR: split wav on PC into <=ChunkSec pieces, run ASR per chunk on device,
# concatenate text and sum wall/decode time. Needed because zipformer-ru crashes on >~20 s.
function Run-AsrChunked {
  Write-Output "--- ASR CHUNKED (<= $ChunkSec s) : $Audio ---"
  $tmp = 'D:\temp\opencode\chunks'
  if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
  New-Item -ItemType Directory -Force -Path $tmp | Out-Null
  $src = "D:\Downloads_D\nuclear_hack\device_bundle\audio\$Audio"
  $splitPy = @'
import sys, os, soundfile as sf
src, outdir, chunk = sys.argv[1], sys.argv[2], float(sys.argv[3])
data, sr = sf.read(src, dtype='float32')
if data.ndim > 1: data = data.mean(axis=1)
n = int(round(chunk * sr))
total = len(data)
parts = []
for i, start in enumerate(range(0, total, n)):
    seg = data[start:start+n]
    p = os.path.join(outdir, 'chunk_%03d.wav' % i)
    sf.write(p, seg, sr, subtype='PCM_16')
    parts.append((p, len(seg)/sr))
print('TOTAL_SEC=%.3f SR=%d CHUNKS=%d' % (total/sr, sr, len(parts)))
for p, d in parts: print('CHUNK %s %.3f' % (os.path.basename(p), d))
'@
  $parsePy = @'
import sys, re
s = open(sys.argv[1], encoding='utf-8', errors='replace').read()
m = re.search(r'Elapsed seconds: ([0-9.]+)', s)
t = re.search(r'"text": "(.*?)", "timestamps"', s, re.S)
print('DEC=%.3f' % (float(m.group(1)) if m else -1))
print('TXT=' + (t.group(1) if t else ''))
'@
  $splitFile = Join-Path $tmp 'split.py'
  $parseFile = Join-Path $tmp 'parse.py'
  Set-Content -Path $splitFile -Value $splitPy -Encoding UTF8
  Set-Content -Path $parseFile -Value $parsePy -Encoding UTF8
  $splitOut = & $py $splitFile $src $tmp $ChunkSec
  $splitOut | ForEach-Object { Write-Output $_ }
  $totalSec = 0.0
  foreach ($line in $splitOut) {
    if ($line -match '^TOTAL_SEC=([0-9.]+)') { $totalSec = [double]$Matches[1] }
  }
  & $adb -s $serial shell "mkdir -p $dest/audio/chunks" | Out-Null
  $chunkFiles = Get-ChildItem $tmp -Filter 'chunk_*.wav' | Sort-Object Name
  foreach ($cf in $chunkFiles) { & $adb -s $serial push $cf.FullName "$dest/audio/chunks/" | Out-Null }
  $sumWallMs = 0.0; $sumDecode = 0.0; $maxHwm = 0; $texts = @()
  foreach ($cf in $chunkFiles) {
    $name = $cf.Name
    $outDev = "$dest/audio/chunks/out_$name.txt"
    $cmd = "cd /data/local/tmp/pipeline && $ldenv ./bench.sh $outDev ./bin/sherpa/sherpa-onnx-offline --tokens=$dest/models/asr/tokens.txt --encoder=$dest/models/asr/encoder.int8.onnx --decoder=$dest/models/asr/decoder.int8.onnx --joiner=$dest/models/asr/joiner.int8.onnx $dest/audio/chunks/$name"
    $bench = & $adb -s $serial shell $cmd
    $benchLine = ($bench | Where-Object { $_ -match 'WALL_MS=' }) -join ' '
    Write-Output ("[$name] " + $benchLine)
    if ($benchLine -match 'WALL_MS=(\d+)') { $sumWallMs += [double]$Matches[1] }
    if ($benchLine -match 'VmHWM_KB=(\d+)') { $h = [double]$Matches[1]; if ($h -gt $maxHwm) { $maxHwm = $h } }
    $localOut = Join-Path $tmp ("out_" + $name + ".txt")
    & $adb -s $serial pull $outDev $localOut | Out-Null
    $parsed = & $py $parseFile $localOut
    foreach ($pl in $parsed) {
      if ($pl -match '^DEC=([0-9.]+)') { $sumDecode += [double]$Matches[1] }
      if ($pl -match '^TXT=(.*)$') { $texts += $Matches[1] }
    }
  }
  $joined = ($texts -join ' ').Trim()
  $wallSec = $sumWallMs / 1000.0
  $wallRtf = if ($totalSec -gt 0) { $wallSec / $totalSec } else { 0 }
  $decRtf = if ($totalSec -gt 0) { $sumDecode / $totalSec } else { 0 }
  $nChunks = $chunkFiles.Count
  $hwmMb = $maxHwm / 1024.0
  Write-Output ("CHUNKED_SUMMARY audio=$Audio total_sec={0:N3} chunks={1} wall_sec={2:N3} wall_rtf={3:N4} decode_sec={4:N3} decode_rtf={5:N4} VmHWM_MB={6:N1}" -f $totalSec, $nChunks, $wallSec, $wallRtf, $sumDecode, $decRtf, $hwmMb)
  Write-Output ("CHUNKED_TEXT: " + $joined)
}

function Run-Diar {
  Write-Output "--- Diarization (pyannote seg + 3dspeaker, thr 0.7) ---"
  & $adb -s $serial shell "cd /data/local/tmp/pipeline && $ldenv ./bin/sherpa/sherpa-onnx-offline-speaker-diarization --segmentation.pyannote-model=$dest/models/diar/pyannote-segmentation-3.0/model.fixed.onnx --embedding.model=$dest/models/diar/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx --clustering.cluster-threshold=0.7 $dest/audio/$Audio"
}
function Run-Llm {
  Write-Output "--- LLM: see reports/profiler/DEVICE_REPORT.md (llm_cli + QVikhr int4) ---"
}
switch ($Stage) {
  'asr'  { if ($Chunked) { Run-AsrChunked } else { Run-Asr } }
  'diar' { Run-Diar }
  'llm'  { Run-Llm }
  'all'  { if ($Chunked) { Run-AsrChunked } else { Run-Asr }; Run-Diar; Run-Llm }
}
$sw.Stop()
Write-Output ("wall_ms=" + $sw.ElapsedMilliseconds)
