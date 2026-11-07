# push_to_device.ps1 — push device_bundle to Samsung SM-N9750.
# Models/audio -> /sdcard/pipeline (readable, noexec); binaries -> /data/local/tmp/pipeline/bin/sherpa (executable).
# Usage: powershell -ExecutionPolicy Bypass -File scripts\push_to_device.ps1 [-SkipLlm]
param([switch]$SkipLlm)
$ErrorActionPreference = 'Stop'
$adb    = 'D:\temp\opencode\platform-tools\adb.exe'
$serial = 'RF8N10DPSDL'
$bundle = Split-Path -Parent $PSScriptRoot
$dest   = '/sdcard/pipeline'
$bin    = '/data/local/tmp/pipeline/bin/sherpa'

& $adb -s $serial shell "mkdir -p $dest/models/asr $dest/models/diar $dest/models/llm $dest/audio $bin"
& $adb -s $serial push "$bundle\models\asr"  "$dest/models/"
& $adb -s $serial push "$bundle\models\diar" "$dest/models/"
if (-not $SkipLlm) { & $adb -s $serial push "$bundle\models\llm" "$dest/models/" }
& $adb -s $serial push "$bundle\audio"       "$dest/"
& $adb -s $serial push "$bundle\bin\sherpa\." "$bin/"
& $adb -s $serial shell "chmod 755 $bin/*; ls -la $bin"
Write-Output "pushed bundle to $dest (binaries: $bin)"
