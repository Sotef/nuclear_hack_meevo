#!/system/bin/sh
# measure_all.sh — 3 runs (1 cold + 2 warm) per file per stage.
cd /data/local/tmp/pipeline || exit 1
export LD_LIBRARY_PATH=/data/local/tmp/pipeline/bin/sherpa
export LD_PRELOAD=/data/local/tmp/pipeline/bin/sherpa/libhashshim.so
ASR="./bin/sherpa/sherpa-onnx-offline --tokens=/sdcard/pipeline/models/asr/tokens.txt --encoder=/sdcard/pipeline/models/asr/encoder.int8.onnx --decoder=/sdcard/pipeline/models/asr/decoder.int8.onnx --joiner=/sdcard/pipeline/models/asr/joiner.int8.onnx"
DIAR="./bin/sherpa/sherpa-onnx-offline-speaker-diarization --segmentation.pyannote-model=/sdcard/pipeline/models/diar/pyannote-segmentation-3.0/model.fixed.onnx --embedding.model=/sdcard/pipeline/models/diar/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx --clustering.cluster-threshold=0.7"
for f in ruls_0001 diar_0002 diar_0003; do
  for r in 1 2 3; do
    res=$(./bench.sh /data/local/tmp/pipeline/out_asr_${f}_r${r}.txt $ASR /sdcard/pipeline/audio/${f}.wav)
    echo "ASR $f r$r $res"
  done
  for r in 1 2 3; do
    res=$(./bench.sh /data/local/tmp/pipeline/out_diar_${f}_r${r}.txt $DIAR /sdcard/pipeline/audio/${f}.wav)
    echo "DIAR $f r$r $res"
  done
done
echo DONE
