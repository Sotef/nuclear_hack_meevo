package com.meevo.app

import android.content.Context
import com.k2fsa.sherpa.onnx.FastClusteringConfig
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineSpeakerDiarization
import com.k2fsa.sherpa.onnx.OfflineSpeakerDiarizationConfig
import com.k2fsa.sherpa.onnx.OfflineSpeakerSegmentationModelConfig
import com.k2fsa.sherpa.onnx.OfflineSpeakerSegmentationPyannoteModelConfig
import com.k2fsa.sherpa.onnx.OfflineTransducerModelConfig
import com.k2fsa.sherpa.onnx.SpeakerEmbeddingExtractorConfig
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.abs
import kotlin.math.min

class Pipeline(private val ctx: Context, private val root: File) {
    private var recognizer: OfflineRecognizer? = null
    private var diarization: OfflineSpeakerDiarization? = null

    data class AsgSeg(val start: Float, val end: Float, val text: String)
    data class DiarSeg(val start: Float, val end: Float, val speaker: Int)
    data class MergedSeg(val start: Float, val end: Float, val text: String, val speaker: Int)

    fun load(onStage: (String) -> Unit) {
        onStage("Загрузка ASR...")
        val asrDir = File(root, "asr")
        // GigaAM v3: 64 mel-бинов (не 80), трансducer-конфиг nemo_transducer
        val feat = FeatureConfig(sampleRate = 16000, featureDim = 64, dither = 0f)
        val modelCfg = OfflineModelConfig(
            transducer = OfflineTransducerModelConfig(
                encoder = File(asrDir, "encoder.int8.onnx").absolutePath,
                decoder = File(asrDir, "decoder.onnx").absolutePath,
                joiner = File(asrDir, "joiner.onnx").absolutePath
            ),
            tokens = File(asrDir, "tokens.txt").absolutePath,
            numThreads = 4,
            debug = false,
            provider = "cpu"
        )
        modelCfg.modelType = "nemo_transducer"
        recognizer = OfflineRecognizer(
            null,
            OfflineRecognizerConfig(
                featConfig = feat,
                modelConfig = modelCfg,
                decodingMethod = "greedy_search"
            )
        )
        onStage("Загрузка диаризации...")
        val diarDir = File(root, "diar")
        val segCfg = OfflineSpeakerSegmentationModelConfig(
            pyannote = OfflineSpeakerSegmentationPyannoteModelConfig(
                model = File(diarDir, "model.fixed.onnx").absolutePath,
                windowShiftRatio = 0.5f
            ),
            numThreads = 2
        )
        val embCfg = SpeakerEmbeddingExtractorConfig(
            model = File(diarDir, "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx").absolutePath,
            numThreads = 2
        )
        val clustCfg = FastClusteringConfig(numClusters = -1, threshold = 0.7f)
        diarization = OfflineSpeakerDiarization(
            null,
            OfflineSpeakerDiarizationConfig(
                segmentation = segCfg,
                embedding = embCfg,
                clustering = clustCfg
            )
        )
    }

    fun asr(
        samples: FloatArray,
        onStage: (String) -> Unit,
        onProgress: ((Int) -> Unit)? = null
    ): List<AsgSeg> {
        val r = recognizer ?: error("ASR not loaded")
        // GigaAM: родной лимит ~200 с, деградация >~90 с → чанки 60 с (из rashifruy)
        val chunkSize = 16000 * 60
        val out = ArrayList<AsgSeg>()
        val totalChunks = (samples.size + chunkSize - 1) / chunkSize
        var off = 0
        var chunkIdx = 0
        while (off < samples.size) {
            val end = min(off + chunkSize, samples.size)
            onStage("ASR: чанк ${chunkIdx + 1}/$totalChunks...")
            onProgress?.invoke(chunkIdx * 100 / totalChunks)
            val part = samples.copyOfRange(off, end)
            val stream = r.createStream()
            stream.acceptWaveform(part, 16000)
            r.decode(stream)
            val res = r.getResult(stream)
            stream.release()
            val base = off / 16000f
            val toks: Array<String>? = res.tokens
            val ts: FloatArray? = res.timestamps
            if (toks == null || ts == null || toks.isEmpty() || ts.isEmpty() || ts.size != toks.size) {
                if (res.text.isNotBlank()) out.add(AsgSeg(base, end / 16000f, res.text.trim()))
            } else {
                var segStart = ts[0]
                var last = ts[0]
                val sb = StringBuilder()
                for (i in toks.indices) {
                    val t = ts[i]
                    if (i > 0 && t - last > 0.7f) {
                        if (sb.isNotBlank()) {
                            out.add(AsgSeg(base + segStart, base + last + 0.3f, sb.toString().trim()))
                        }
                        sb.setLength(0)
                        segStart = t
                    }
                    sb.append(toks[i].replace("\u2581", " "))
                    last = t
                }
                if (sb.isNotBlank()) {
                    out.add(AsgSeg(base + segStart, base + min(last + 0.3f, end / 16000f - base), sb.toString().trim()))
                }
            }
            off = end
            chunkIdx++
        }
        onProgress?.invoke(100)
        return out
    }

    fun diarize(
        samples: FloatArray,
        onStage: (String) -> Unit,
        onProgress: ((Int) -> Unit)? = null
    ): List<DiarSeg> {
        onStage("Диаризация...")
        val d = diarization ?: return emptyList()
        if (onProgress != null) {
            val prog = onProgress
            // Реальный прогресс: sherpa вызывает callback (обработано кадров, всего, мс)
            val segs = d.processWithCallback(samples, { processed, total, _ ->
                val t = total ?: 1
                val p = processed ?: 0
                if (t > 0) prog?.invoke((p.toLong() * 100 / t).toInt().coerceIn(0, 100))
                0
            }, 0L)
            prog?.invoke(100)
            return segs.map { DiarSeg(it.start, it.end, it.speaker) }
        }
        onProgress?.invoke(50)
        val segs = d.process(samples)
        onProgress?.invoke(100)
        return segs.map { DiarSeg(it.start, it.end, it.speaker) }
    }

    fun merge(asrSegs: List<AsgSeg>, diarSegs: List<DiarSeg>): List<MergedSeg> {
        return asrSegs.map { s ->
            var best = -1
            var bestOv = 0f
            for (d in diarSegs) {
                val ov = min(s.end, d.end) - Math.max(s.start, d.start)
                if (ov > bestOv) {
                    bestOv = ov
                    best = d.speaker
                }
            }
            if (best < 0 && diarSegs.isNotEmpty()) {
                val nd = diarSegs.minByOrNull {
                    min(abs(it.start - s.start), abs(it.end - s.end))
                }
                if (nd != null) best = nd.speaker
            }
            MergedSeg(s.start, s.end, s.text, if (best < 0) 0 else best)
        }
    }

    fun toJson(
        audioId: String,
        durationSec: Float,
        merged: List<MergedSeg>,
        punctuator: ((String) -> String)?,
        onStage: ((String) -> Unit)? = null,
        onProgress: ((Int) -> Unit)? = null
    ): JSONObject {
        val speakers = merged.map { it.speaker }.distinct().sorted()
        val segArr = JSONArray()
        for ((idx, m) in merged.withIndex()) {
            onProgress?.invoke(idx * 100 / maxOf(1, merged.size))
            val raw = m.text
            var text = raw
            if (punctuator != null && raw.isNotBlank()) {
                onStage?.invoke("Пунктуация: сегмент ${idx + 1}/${merged.size}...")
                try {
                    text = punctuator(raw)
                } catch (_: Exception) {
                    text = raw
                }
            }
            val o = JSONObject()
            o.put("speaker", "SPEAKER_" + (m.speaker + 1))
            o.put("start", Math.round(m.start * 100.0) / 100.0)
            o.put("end", Math.round(m.end * 100.0) / 100.0)
            o.put("text", text)
            o.put("text_raw", raw)
            o.put("doubtful", false)
            val edits = JSONArray()
            if (text != raw) {
                edits.put(JSONObject().put("from", raw).put("to", text).put("reason", "punct"))
            }
            o.put("edits", edits)
            o.put("confidence", 0.0)
            segArr.put(o)
        }
        onProgress?.invoke(100)
        val spArr = JSONArray()
        for (s in speakers) {
            spArr.put(JSONObject().put("id", "SPEAKER_" + (s + 1)).put("label", JSONObject.NULL))
        }
        val meta = JSONObject()
            .put("asr_model", "GigaAM-v3-rnnt int8 (sherpa-onnx, nemo_transducer)")
            .put("diar_model", "campplus+pyannote-seg-3.0+FastClustering(thr=0.7)")
            .put("punct_model", if (punctuator != null) "RUPunct-small int8 (t9)" else "none")
            .put("llm_model", "none")
            .put("postproc_version", "postproc_v1 (punctuate)")
            .put("version", "meevo-0.3")
        return JSONObject()
            .put("audio_id", audioId)
            .put("duration_sec", Math.round(durationSec * 1000.0) / 1000.0)
            .put("language", "ru")
            .put("speakers", spArr)
            .put("segments", segArr)
            .put("summary", JSONObject.NULL)
            .put("chapters", JSONArray())
            .put("meta", meta)
    }

    fun release() {
        recognizer?.release()
        diarization?.release()
        recognizer = null
        diarization = null
    }
}
