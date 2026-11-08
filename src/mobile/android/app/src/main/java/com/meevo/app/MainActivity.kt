package com.meevo.app

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.util.TypedValue
import android.view.View
import android.view.animation.AlphaAnimation
import android.view.animation.Animation
import android.view.animation.AnimationSet
import android.view.animation.ScaleAnimation
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.documentfile.provider.DocumentFile
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : Activity() {

    // ---------- Цвета — из темы Material 3 (themes.xml / colors.xml) ----------

    companion object {
        private const val REQ_EXPORT = 8

        private var pipeline: Pipeline? = null
        private var punctEngine: PunctEngine? = null
        private var llm: LlmEngine? = null
        private var modelsReady = false
        private var lastJson: JSONObject? = null
        private var lastAudioId: String? = null
        private var currentFile: File? = null
        private var merged: List<Pipeline.MergedSeg> = emptyList()
    }

    // ---------- View ----------
    private lateinit var rootBox: LinearLayout
    private lateinit var titleText: TextView
    private lateinit var metaText: TextView
    private lateinit var screenRecord: LinearLayout
    private lateinit var screenProgress: LinearLayout
    private lateinit var screenResult: LinearLayout
    private lateinit var recordButton: Button
    private lateinit var fileButton: Button
    private lateinit var runButton: Button
    private lateinit var sourceLabel: TextView
    private lateinit var recTimer: TextView
    private lateinit var recDot: View
    private val rings = ArrayList<View>()
    private lateinit var etaText: TextView
    private lateinit var stageText: TextView
    private lateinit var overallBar: ProgressBar
    private lateinit var overallPct: TextView
    private lateinit var stagesBox: LinearLayout
    private lateinit var tabText: Button
    private lateinit var tabSummary: Button
    private lateinit var tabChapters: Button
    private lateinit var panelText: LinearLayout
    private lateinit var panelSummary: LinearLayout
    private lateinit var panelChapters: LinearLayout
    private lateinit var segmentsBox: LinearLayout
    private lateinit var chaptersBox: LinearLayout
    private lateinit var summaryText: TextView
    private lateinit var playButton: Button
    private lateinit var seekBar: SeekBar
    private lateinit var timeText: TextView
    private lateinit var exportButton: Button
    private lateinit var exportText: TextView
    private lateinit var backButton: Button

    // ---------- Этапы и ETA ----------
    private data class StageUi(
        val name: String,
        val icon: TextView,
        val bar: ProgressBar,
        val pct: TextView,
        val eta: TextView
    )

    private val stageOrder = listOf("Декод", "ASR", "Диаризация", "Пунктуация", "LLM-анализ")
    private val stageWeight = mapOf(
        "Декод" to 5, "ASR" to 35, "Диаризация" to 40, "Пунктуация" to 10, "LLM-анализ" to 10
    )
    private val stageUi = LinkedHashMap<String, StageUi>()
    private val stagePct = HashMap<String, Int>()
    private val stageStartMs = HashMap<String, Long>()
    private val stageEtaSec = HashMap<String, Double>()
    private var audioDurSec = 0f
    private var runStartMs = 0L

    private val demoWavs = listOf("demo_long.wav", "ruls_0001.wav", "diar_0002.wav", "diar_0003.wav")

    private var mediaPlayer: MediaPlayer? = null
    private val handler = Handler(Looper.getMainLooper())
    private val ticker = object : Runnable {
        override fun run() {
            val mp = mediaPlayer ?: return
            if (mp.isPlaying) {
                seekBar.progress = mp.currentPosition
                timeText.text = fmtTime(mp.currentPosition.toLong())
            }
            handler.postDelayed(this, 300)
        }
    }

    private var recorder: AudioRecord? = null
    private val recording = AtomicBoolean(false)
    private var recordStartMs = 0L
    private val recTicker = object : Runnable {
        override fun run() {
            val sec = (System.currentTimeMillis() - recordStartMs) / 1000
            recTimer.text = String.format(Locale.US, "%02d:%02d", sec / 60, sec % 60)
            handler.postDelayed(this, 500)
        }
    }

    // ---------- Жизненный цикл ----------
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(resources.getIdentifier("activity_main", "layout", packageName))
        bindViews()
        setupClicks()
        updateSourceLabel()

        Thread {
            if (modelsReady) {
                ui { restoreState() }
            } else {
                prepareModels()
            }
        }.start()
    }

    override fun onDestroy() {
        super.onDestroy()
        handler.removeCallbacks(ticker)
        handler.removeCallbacks(recTicker)
        mediaPlayer?.release()
        mediaPlayer = null
    }

    private fun bindViews() {
        rootBox = find("rootBox")
        titleText = find("titleText")
        metaText = find("metaText")
        screenRecord = find("screenRecord")
        screenProgress = find("screenProgress")
        screenResult = find("screenResult")
        recordButton = find("recordButton")
        fileButton = find("fileButton")
        runButton = find("runButton")
        sourceLabel = find("sourceLabel")
        recTimer = find("recTimer")
        recDot = find("recDot")
        rings.add(find("ring1")); rings.add(find("ring2")); rings.add(find("ring3"))
        etaText = find("etaText")
        stageText = find("stageText")
        overallBar = find("overallBar")
        overallPct = find("overallPct")
        stagesBox = find("stagesBox")
        tabText = find("tabText")
        tabSummary = find("tabSummary")
        tabChapters = find("tabChapters")
        panelText = find("panelText")
        panelSummary = find("panelSummary")
        panelChapters = find("panelChapters")
        segmentsBox = find("segmentsBox")
        chaptersBox = find("chaptersBox")
        summaryText = find("summaryText")
        playButton = find("playButton")
        seekBar = find("seekBar")
        timeText = find("timeText")
        exportButton = find("exportButton")
        exportText = find("exportText")
        backButton = find("backButton")
    }

    private fun <T : View> find(name: String): T =
        findViewById(resources.getIdentifier(name, "id", packageName))

    private fun dp(v: Int) = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), resources.displayMetrics
    ).toInt()

    // ---------- Анимации ----------
    private fun startRecordPulse() {
        rings.forEachIndexed { i, ring ->
            val animSet = AnimationSet(true)
            val sc = ScaleAnimation(1f, 2.2f, 1f, 2.2f, Animation.RELATIVE_TO_SELF, .5f, Animation.RELATIVE_TO_SELF, .5f)
            val al = AlphaAnimation(.5f, 0f)
            animSet.addAnimation(sc)
            animSet.addAnimation(al)
            animSet.setDuration(1800)
            animSet.setStartOffset(i * 600L)
            animSet.setRepeatCount(Animation.INFINITE)
            ring.startAnimation(animSet)
        }
        val blink = AlphaAnimation(1f, .2f)
        blink.setDuration(500)
        blink.setRepeatMode(Animation.REVERSE)
        blink.setRepeatCount(Animation.INFINITE)
        recDot.startAnimation(blink)
    }

    private fun stopRecordPulse() {
        rings.forEach { it.clearAnimation() }
        recDot.clearAnimation()
    }

    private fun flashSuccess(btn: Button, label: String) {
        val orig = btn.backgroundTintList
        btn.backgroundTintList = ColorStateList.valueOf(getColor(R.color.colorPrimary))
        btn.text = "✓"
        btn.animate().scaleX(.96f).scaleY(.96f).setDuration(100).withEndAction {
            btn.animate().scaleX(1f).scaleY(1f).setDuration(200).start()
        }.start()
        handler.postDelayed({
            btn.text = label
            btn.backgroundTintList = orig
        }, 1400)
    }

    private fun enterResult() {
        val views = listOf<View>(tabText, tabSummary, tabChapters, seekBar, playButton, panelText, exportButton)
        views.forEachIndexed { i, v ->
            v.alpha = 0f
            v.translationY = dp(40).toFloat()
            v.animate().alpha(1f).translationY(0f).setDuration(800).setStartDelay(200L + i * 60L).start()
        }
    }

    // ---------- Этапы + ETA ----------
    private fun buildStageRows() {
        stagesBox.removeAllViews()
        stageUi.clear()
        stagePct.clear()
        stageStartMs.clear()
        stageEtaSec.clear()
        for (name in stageOrder) {
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = android.view.Gravity.CENTER_VERTICAL
                setPadding(0, dp(6), 0, dp(6))
            }
            val icon = TextView(this).apply {
                text = "○"
                width = dp(28)
                textSize = 16f
                setTextColor(getColor(R.color.colorPrimary))
            }
            val label = TextView(this).apply {
                text = name
                width = dp(96)
                setTextColor(getColor(R.color.colorOnSurfaceVariant))
                textSize = 13f
            }
            val bar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
                max = 100
                progressTintList = ColorStateList.valueOf(getColor(R.color.colorPrimary))
            }
            val pct = TextView(this).apply {
                text = "0%"
                width = dp(44)
                gravity = android.view.Gravity.END
                setTextColor(getColor(R.color.colorOnSurfaceVariant))
                textSize = 12f
            }
            val eta = TextView(this).apply {
                text = ""
                width = dp(64)
                gravity = android.view.Gravity.END
                setTextColor(getColor(R.color.colorOnSurfaceVariant))
                textSize = 12f
            }
            row.addView(icon)
            row.addView(label)
            row.addView(bar, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
            row.addView(pct)
            row.addView(eta)
            stagesBox.addView(row)
            stageUi[name] = StageUi(name, icon, bar, pct, eta)
            stagePct[name] = 0
        }
    }

    /** Оценка длительности этапа (сек) по длительности аудио — из замеров на устройстве. */
    private fun estimateStage(name: String, dur: Float): Double = when (name) {
        "Декод" -> 0.05 * dur + 1.0
        "ASR" -> 0.10 * dur
        "Диаризация" -> 0.37 * dur
        "Пунктуация" -> 0.01 * dur + 1.0
        else -> 30.0 + 0.25 * dur
    }

    private fun setStage(name: String, percent: Int) {
        ui {
            val p = percent.coerceIn(0, 100)
            if (!stageStartMs.containsKey(name)) stageStartMs[name] = System.currentTimeMillis()
            stagePct[name] = p
            val st = stageUi[name] ?: return@ui
            st.bar.progress = p
            st.pct.text = "$p%"
            st.icon.text = if (p >= 100) "✓" else "●"
            val elapsed = (System.currentTimeMillis() - (stageStartMs[name] ?: System.currentTimeMillis())) / 1000.0
            val est = if (p >= 5) elapsed / p * (100 - p) else estimateStage(name, audioDurSec)
            stageEtaSec[name] = if (p >= 100) 0.0 else est
            st.eta.text = if (p >= 100) "готово" else "≈ " + fmtSec(est)

            var total = 0
            var wsum = 0
            for ((k, w) in stageWeight) {
                total += (stagePct[k] ?: 0) * w
                wsum += w
            }
            val overall = total / wsum
            overallBar.progress = overall
            overallPct.text = "$overall%"

            val remaining = stageOrder.sumOf { stageEtaSec[it] ?: estimateStage(it, audioDurSec) }
            val done = stageOrder.count { (stagePct[it] ?: 0) >= 100 }
            etaText.text = if (overall >= 100) "Готово"
            else "Осталось ~" + fmtSec(remaining) + " · этап " + (done + 1) + "/" + stageOrder.size
        }
    }

    private fun fmtSec(sec: Double): String {
        val s = sec.toLong().coerceAtLeast(0)
        return String.format(Locale.US, "%d:%02d", s / 60, s % 60)
    }

    private fun showScreen(which: String) {
        ui {
            screenRecord.visibility = if (which == "record") View.VISIBLE else View.GONE
            screenProgress.visibility = if (which == "progress") View.VISIBLE else View.GONE
            screenResult.visibility = if (which == "result") View.VISIBLE else View.GONE
        }
    }

    // ---------- Источник ----------
    private fun updateSourceLabel() {
        val f = currentFile
        sourceLabel.text = "Источник: " + (f?.name ?: "demo_long.wav")
    }

    private fun ui(f: () -> Unit) = runOnUiThread(f)

    private fun setStatus(msg: String) {
        Log.d("Meevo", "status: $msg")
        ui { metaText.text = msg }
    }

    private fun prepareModels() {
        if (modelsReady) return
        try {
            val root = getExternalFilesDir("models") ?: filesDir
            val ready = File(root, ".ready_v3")
            if (!ready.exists()) {
                ui { metaText.text = "Распаковка моделей (первый запуск)..." }
                copyAssetDir("models", root)
                ready.writeText("ok")
            }
            val cfgDir = getExternalFilesDir("config") ?: filesDir
            cfgDir.mkdirs()
            val promptFile = File(cfgDir, "system_prompt.txt")
            if (!promptFile.exists()) {
                assets.open("prompts/system_prompt_ru.txt").use { input ->
                    promptFile.outputStream().use { output -> input.copyTo(output) }
                }
            }
            val p = Pipeline(this, root)
            p.load { setStatus(it) }
            pipeline = p
            // Пунктуация (onnxruntime) несовместима с sherpa-onnx по символьным
            // версиям libonnxruntime.so (VERS_1.23.0 vs VERS_1.28.2) — делаем её
            // опциональной, чтобы ASR+диаризация работали независимо.
            try {
                punctEngine = PunctEngine(File(root, "punct"))
            } catch (e: Throwable) {
                Log.e("Meevo", "punct disabled: " + e.message)
                punctEngine = null
            }
            modelsReady = true
            ui {
                metaText.text = "Офлайн · модели загружены"
                runButton.isEnabled = true
            }
        } catch (e: Exception) {
            Log.e("Meevo", "prepareModels failed", e)
            ui { metaText.text = "Ошибка подготовки: " + e.message }
        }
    }

    private fun copyAssetDir(assetPath: String, outDir: File) {
        val am = assets
        val children = am.list(assetPath) ?: return
        if (children.isEmpty()) return
        for (child in children) {
            val a = "$assetPath/$child"
            val sub = am.list(a)
            if (sub == null || sub.isEmpty()) {
                val outFile = File(outDir, child)
                outFile.parentFile?.mkdirs()
                am.open(a).use { input ->
                    outFile.outputStream().use { output -> input.copyTo(output) }
                }
                Log.d("Meevo", "copied $a -> ${outFile.length()} bytes")
            } else {
                File(outDir, child).mkdirs()
                copyAssetDir(a, File(outDir, child))
            }
        }
    }

    private fun restoreState() {
        runButton.isEnabled = true
        metaText.text = "Офлайн · модели загружены (восстановлено)"
        val json = lastJson
        if (json != null) {
            showScreen("result")
            renderSegments(json)
            exportButton.isEnabled = true
            val ch = json.optJSONArray("chapters")
            if (ch != null && ch.length() > 0) renderChaptersFromJson(ch)
            val sum = json.optString("summary", "")
            if (sum.isNotEmpty()) summaryText.text = sum
            currentFile?.let { if (it.exists()) setupPlayer(it) }
        }
    }

    private fun toggleRecord() {
        if (recording.get()) {
            recording.set(false)
            recordButton.text = "Микрофон"
            stopRecordPulse()
            handler.removeCallbacks(recTicker)
            return
        }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 42)
            return
        }
        val sampleRate = 16000
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        val rec = AudioRecord(
            MediaRecorder.AudioSource.MIC, sampleRate,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBuf, 16000 * 2)
        )
        recorder = rec
        recording.set(true)
        recordButton.text = "Стоп"
        recordStartMs = System.currentTimeMillis()
        recTimer.text = "00:00"
        startRecordPulse()
        handler.post(recTicker)
        setStatus("Запись с микрофона...")
        Thread {
            val chunks = ArrayList<ByteArray>()
            rec.startRecording()
            val buf = ByteArray(32000)
            while (recording.get()) {
                val n = rec.read(buf, 0, buf.size)
                if (n > 0) chunks.add(buf.copyOf(n))
            }
            rec.stop()
            rec.release()
            recorder = null
            var total = 0
            for (c in chunks) total += c.size
            val pcm = ByteArray(total)
            var pos = 0
            for (c in chunks) {
                System.arraycopy(c, 0, pcm, pos, c.size)
                pos += c.size
            }
            val samples = FloatArray(pcm.size / 2)
            for (i in samples.indices) {
                val lo = pcm[i * 2].toInt() and 0xff
                val hi = pcm[i * 2 + 1].toInt()
                samples[i] = ((hi shl 8) or lo).toShort() / 32768f
            }
            val name = "mic_" + System.currentTimeMillis() + ".wav"
            val f = File(cacheDir, name)
            WavIO.write(f, samples, sampleRate)
            ui {
                currentFile = f
                updateSourceLabel()
                metaText.text = "Записано " + String.format(Locale.US, "%.1f", samples.size / 16000f) + " c"
            }
        }.start()
    }

    private fun pickFile() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "audio/*"
        }
        startActivityForResult(intent, 7)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 7 && resultCode == RESULT_OK && data != null) {
            val uri = data.data ?: return
            Thread {
                try {
                    val name = "file_" + System.currentTimeMillis() + ".bin"
                    val f = File(cacheDir, name)
                    contentResolver.openInputStream(uri).use { input ->
                        f.outputStream().use { output -> input!!.copyTo(output) }
                    }
                    ui {
                        currentFile = f
                        updateSourceLabel()
                        metaText.text = "Файл загружен: " + uri.lastPathSegment
                    }
                } catch (e: Exception) {
                    ui { metaText.text = "Ошибка файла: " + e.message }
                }
            }.start()
        } else if (requestCode == REQ_EXPORT && resultCode == RESULT_OK && data?.data != null) {
            val treeUri = data.data!!
            Thread {
                val tree = DocumentFile.fromTreeUri(this, treeUri)
                if (tree == null) {
                    ui { exportText.text = "Не удалось открыть выбранную папку" }
                    return@Thread
                }
                try {
                    val count = writeExports(tree)
                    ui {
                        exportText.text = "Экспортировано файлов: $count (JSON, SRT, TXT, схема)"
                        flashSuccess(exportButton, "Экспорт готов")
                    }
                } catch (e: Exception) {
                    ui { exportText.text = "Ошибка экспорта: " + e.message }
                }
            }.start()
        }
    }

    private fun resolveSourceFile(): File? {
        val f = currentFile
        if (f != null && f.exists()) return f
        val out = File(cacheDir, "demo_long.wav")
        if (!out.exists()) assets.open("audio/demo_long.wav").use { input ->
            out.outputStream().use { output -> input.copyTo(output) }
        }
        return out
    }

    // ---------- ПОЛНЫЙ ПАЙПЛАЙН: одна кнопка ----------
    private fun runPipeline() {
        val p = pipeline ?: return
        if (!modelsReady) return
        if (recording.get()) {
            Toast.makeText(this, "Остановите запись", Toast.LENGTH_SHORT).show()
            return
        }
        val f = resolveSourceFile()
        if (f == null || !f.exists()) {
            Toast.makeText(this, "Источник не выбран", Toast.LENGTH_SHORT).show()
            return
        }
        currentFile = f
        runButton.isEnabled = false
        exportButton.isEnabled = false
        buildStageRows()
        showScreen("progress")
        segmentsBox.removeAllViews()
        chaptersBox.removeAllViews()
        summaryText.text = ""
        runStartMs = System.currentTimeMillis()
        val audioId = f.nameWithoutExtension
        Thread {
            try {
                setStage("Декод", 10)
                ui { stageText.text = "Декодирование аудио..." }
                val samples = AudioDecoder.decodeToMono16k(f, 16000)
                if (samples.isEmpty() || samples.size < 8000) {
                    throw IllegalArgumentException("Не удалось декодировать аудио или файл слишком короткий")
                }
                audioDurSec = samples.size / 16000f
                setStage("Декод", 100)

                val asrSegs = p.asr(samples, { ui { stageText.text = it } }) { setStage("ASR", it) }
                val diarSegs = p.diarize(samples, { ui { stageText.text = it } }) { setStage("Диаризация", it) }

                ui { stageText.text = "Сборка реплик..." }
                merged = p.merge(asrSegs, diarSegs)

                val pe = punctEngine
                val punctFn: ((String) -> String)? = if (pe != null) {
                    val fn: (String) -> String = { s -> pe.punctuate(s) }
                    fn
                } else null
                val json = p.toJson(audioId, audioDurSec, merged, punctFn,
                    { ui { stageText.text = it } }) { setStage("Пунктуация", it) }
                lastJson = json
                lastAudioId = audioId

                // LLM-анализ: склейка разрезанных фраз → главы → конспект
                try {
                    val e = getLlm()
                    setStage("LLM-анализ", 5)
                    ui { stageText.text = "LLM: склейка разрезанных фраз..." }
                    val merges = e.findMerges(buildDialog())
                    applyMerges(merges)
                    setStage("LLM-анализ", 35)

                    ui { stageText.text = "LLM: главы..." }
                    val chapters = e.chapters(buildDialog())
                    ui { renderChapters(chapters) }
                    setStage("LLM-анализ", 70)

                    ui { stageText.text = "LLM: конспект..." }
                    val summary = e.summarize(buildDialog())
                    ui { summaryText.text = summary }
                    lastJson?.put("summary", summary)
                    lastJson?.optJSONObject("meta")
                        ?.put("llm_model", "Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning int4")
                    setStage("LLM-анализ", 100)
                } catch (e: Throwable) {
                    Log.e("Meevo", "llm analysis failed", e)
                    setStage("LLM-анализ", 100)
                    ui { stageText.text = "LLM-анализ пропущен: " + e.message }
                }

                val totalMs = System.currentTimeMillis() - runStartMs
                val rtf = totalMs / 1000f / audioDurSec
                ui {
                    showScreen("result")
                    enterResult()
                    renderSegments(json)
                    setupPlayer(f)
                    exportButton.isEnabled = true
                    runButton.isEnabled = true
                    metaText.text = "Готово: " + merged.size + " сегментов, " +
                        String.format(Locale.US, "%.1f", audioDurSec) + " c · RTF " +
                        String.format(Locale.US, "%.2f", rtf)
                }
            } catch (e: Exception) {
                Log.e("Meevo", "runPipeline failed", e)
                ui {
                    showScreen("record")
                    runButton.isEnabled = true
                    metaText.text = "Ошибка: " + e.message
                }
            }
        }.start()
    }

    private fun getLlm(): LlmEngine {
        llm?.let { return it }
        val promptFile = File(getExternalFilesDir("config"), "system_prompt.txt")
        val sysPrompt = if (promptFile.exists()) promptFile.readText()
        else assets.open("prompts/system_prompt_ru.txt").bufferedReader().readText()
        val e = LlmEngine(File(getExternalFilesDir("models"), "llm").absolutePath, sysPrompt)
        llm = e
        return e
    }

    // ---------- Диалог для LLM ----------
    private fun buildDialog(): String {
        val sb = StringBuilder()
        val segs = lastJson?.optJSONArray("segments")
        if (segs != null) {
            for (i in 0 until segs.length()) {
                val s = segs.getJSONObject(i)
                sb.append(fmtTime((s.getDouble("start") * 1000).toLong()))
                    .append(' ').append(s.getString("speaker"))
                    .append(": ").append(s.getString("text")).append('\n')
            }
        } else {
            for (m in merged) {
                sb.append(fmtTime((m.start * 1000).toLong()))
                    .append(' ').append("SPEAKER_").append(m.speaker + 1)
                    .append(": ").append(m.text).append('\n')
            }
        }
        return sb.toString().trim()
    }

    // ---------- Склейка ----------
    private fun applyMerges(raw: String): Int {
        val segs = lastJson?.optJSONArray("segments") ?: return 0
        val pairs = ArrayList<Pair<Int, Int>>()
        try {
            val arr = parseChapters(raw)
            for (i in 0 until arr.length()) {
                val o = arr.getJSONObject(i)
                val a = o.optInt("first", 0) - 1
                val b = o.optInt("second", 0) - 1
                if (a < 0 || b < 0 || a >= segs.length() || b >= segs.length()) continue
                val lo = minOf(a, b)
                val hi = maxOf(a, b)
                if (hi != lo + 1) continue
                pairs.add(lo to hi)
            }
        } catch (e: Exception) {
            Log.e("Meevo", "merges parse failed", e)
            return 0
        }
        if (pairs.isEmpty()) return 0

        val list = ArrayList<JSONObject>()
        for (i in 0 until segs.length()) list.add(segs.getJSONObject(i))
        val used = BooleanArray(list.size)
        val dropped = HashSet<Int>()
        var joined = 0
        for ((lo, hi) in pairs.sortedByDescending { it.second }) {
            if (used[lo] || used[hi]) continue
            val s1 = list[lo]
            val s2 = list[hi]
            val d1 = s1.getDouble("end") - s1.getDouble("start")
            val d2 = s2.getDouble("end") - s2.getDouble("start")
            val sp1 = s1.getString("speaker")
            val sp2 = s2.getString("speaker")
            val cross = sp1 != sp2
            val m = JSONObject()
            m.put("speaker", if (d1 >= d2) sp1 else sp2)
            m.put("start", s1.getDouble("start"))
            m.put("end", s2.getDouble("end"))
            m.put("text", s1.getString("text").trim() + " " + s2.getString("text").trim())
            m.put("text_raw", s1.getString("text_raw").trim() + " " + s2.getString("text_raw").trim())
            m.put("doubtful", cross || s1.optBoolean("doubtful") || s2.optBoolean("doubtful"))
            val edits = JSONArray()
            edits.put(
                JSONObject()
                    .put("from", s1.getString("text_raw") + " | " + s2.getString("text_raw"))
                    .put("to", m.getString("text"))
                    .put("reason", if (cross) "join_cross_speaker" else "join")
            )
            m.put("edits", edits)
            m.put("confidence", 0.0)
            list[lo] = m
            used[lo] = true
            used[hi] = true
            dropped.add(hi)
            joined++
        }
        val newArr = JSONArray()
        for (i in list.indices) {
            if (i !in dropped) newArr.put(list[i])
        }
        lastJson?.put("segments", newArr)
        lastJson?.optJSONObject("meta")?.put("segments_joined", joined)
        return joined
    }

    // ---------- Главы ----------
    private fun parseChapters(raw: String): JSONArray {
        var s = raw.trim()
        s = s.replace(Regex("^```(json)?"), "").replace(Regex("```$"), "").trim()
        val a = s.indexOf('[')
        val b = s.lastIndexOf(']')
        if (a >= 0 && b > a) s = s.substring(a, b + 1)
        return JSONArray(s)
    }

    private fun parseTimeSec(t: String): Double {
        val parts = t.trim().split(":")
        return when (parts.size) {
            3 -> parts[0].toDouble() * 3600 + parts[1].toDouble() * 60 + parts[2].toDouble()
            2 -> parts[0].toDouble() * 60 + parts[1].toDouble()
            else -> t.trim().toDoubleOrNull() ?: 0.0
        }
    }

    private fun renderChapters(raw: String) {
        try {
            renderChaptersFromJson(parseChapters(raw))
        } catch (e: Exception) {
            Log.e("Meevo", "chapters parse failed", e)
            chaptersBox.removeAllViews()
            val tv = TextView(this)
            tv.text = "Не удалось разобрать главы: " + e.message
            chaptersBox.addView(tv)
        }
    }

    private fun renderChaptersFromJson(arr: JSONArray) {
        chaptersBox.removeAllViews()
        val outArr = JSONArray()
        for (i in 0 until arr.length()) {
            val c = arr.getJSONObject(i)
            val startSec = parseTimeSec(c.optString("start", "0"))
            val endSec = parseTimeSec(c.optString("end", "0"))
            val title = c.optString("title", "Блок " + (i + 1))
            outArr.put(
                JSONObject()
                    .put("start", startSec)
                    .put("end", endSec)
                    .put("title", title)
            )
            val tv = TextView(this)
            tv.text = String.format(Locale.US, "[%s] %s", fmtTime((startSec * 1000).toLong()), title)
            tv.textSize = 14f
            tv.setTextColor(getColor(R.color.colorOnSurface))
            tv.setPadding(0, 12, 0, 12)
            tv.setOnClickListener {
                seekTo((startSec * 1000).toInt())
                metaText.text = "Переход: " + title
            }
            chaptersBox.addView(tv)
        }
        lastJson?.put("chapters", outArr)
    }

    // ---------- Плеер ----------
    private fun setupPlayer(f: File) {
        handler.removeCallbacks(ticker)
        mediaPlayer?.release()
        val mp = MediaPlayer()
        try {
            mp.setDataSource(f.absolutePath)
            mp.prepare()
            mediaPlayer = mp
            seekBar.max = mp.duration
            seekBar.progress = 0
            seekBar.isEnabled = true
            playButton.isEnabled = true
            playButton.text = "▶"
            timeText.text = fmtTime(0)
        } catch (e: Exception) {
            Log.e("Meevo", "player failed", e)
        }
    }

    private fun togglePlay() {
        val mp = mediaPlayer ?: return
        if (mp.isPlaying) {
            mp.pause()
            playButton.text = "▶"
        } else {
            mp.start()
            playButton.text = "❚❚"
            handler.post(ticker)
        }
    }

    private fun seekTo(ms: Int) {
        val mp = mediaPlayer ?: return
        mp.seekTo(ms)
        seekBar.progress = ms
        timeText.text = fmtTime(ms.toLong())
        if (!mp.isPlaying) {
            mp.start()
            playButton.text = "❚❚"
            handler.post(ticker)
        }
    }

    private fun fmtTime(ms: Long): String {
        val totalSec = ms / 1000
        return String.format(Locale.US, "%d:%02d", totalSec / 60, totalSec % 60)
    }

    // ---------- Рендер ----------
    private fun renderSegments(json: JSONObject) {
        segmentsBox.removeAllViews()
        val segs = json.getJSONArray("segments")
        for (i in 0 until segs.length()) {
            val s = segs.getJSONObject(i)
            val start = (s.getDouble("start") * 1000).toInt()
            val spk = s.getString("speaker")
            val tv = TextView(this)
            tv.text = String.format(
                Locale.US, "%s · %s\n%s",
                spk, fmtTime(start.toLong()), s.getString("text")
            )
            tv.textSize = 13f
            tv.setTextColor(getColor(R.color.colorOnSurface))
            tv.setPadding(0, 10, 0, 10)
            if (s.optBoolean("doubtful")) tv.text = "⚠ " + tv.text
            tv.setOnClickListener {
                seekTo(start)
                metaText.text = "Переход: " + fmtTime(start.toLong())
            }
            segmentsBox.addView(tv)
        }
    }

    private fun showTab(tab: String) {
        panelText.visibility = if (tab == "text") View.VISIBLE else View.GONE
        panelSummary.visibility = if (tab == "summary") View.VISIBLE else View.GONE
        panelChapters.visibility = if (tab == "chapters") View.VISIBLE else View.GONE
    }

    // ---------- Экспорт (одна кнопка — все форматы, выбор папки через SAF) ----------
    private fun exportAll() {
        if (lastJson == null || lastAudioId == null) return
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT_TREE)
        try {
            startActivityForResult(intent, REQ_EXPORT)
        } catch (e: Exception) {
            ui { exportText.text = "Не удалось открыть выбор папки: " + e.message }
        }
    }

    private fun writeExports(dir: DocumentFile): Int {
        val json = lastJson ?: return 0
        val id = lastAudioId ?: return 0
        var n = 0
        fun write(name: String, mime: String, content: String) {
            val f = dir.createFile(mime, name)
                ?: throw java.io.IOException("Не удалось создать файл $name")
            contentResolver.openOutputStream(f.uri)?.use { os ->
                os.write(content.toByteArray(Charsets.UTF_8))
            } ?: throw java.io.IOException("Не удалось записать файл $name")
            n++
        }
        write("$id.json", "application/json", json.toString(2))
        write("$id.srt", "application/x-subrip", buildSrt())
        write("$id.txt", "text/plain", buildTxt())
        write("$id.schema.json", "application/json", buildSchema())
        return n
    }

    private fun buildSrt(): String {
        val json = lastJson ?: return ""
        val segs = json.getJSONArray("segments")
        val sb = StringBuilder()
        for (i in 0 until segs.length()) {
            val s = segs.getJSONObject(i)
            sb.append(i + 1).append('\n')
                .append(srtTime(s.getDouble("start"))).append(" --> ").append(srtTime(s.getDouble("end"))).append('\n')
                .append(s.getString("speaker")).append(": ").append(s.getString("text")).append("\n\n")
        }
        return sb.toString()
    }

    private fun srtTime(sec: Double): String {
        val ms = (sec * 1000).toLong()
        return String.format(
            Locale.US, "%02d:%02d:%02d,%03d",
            ms / 3600000, (ms / 60000) % 60, (ms / 1000) % 60, ms % 1000
        )
    }

    private fun buildTxt(): String {
        val json = lastJson ?: return ""
        val segs = json.getJSONArray("segments")
        val sb = StringBuilder()
        for (i in 0 until segs.length()) {
            val s = segs.getJSONObject(i)
            sb.append(fmtTime((s.getDouble("start") * 1000).toLong()))
                .append("  ").append(s.getString("speaker"))
                .append(": ").append(s.getString("text")).append('\n')
        }
        return sb.toString()
    }

    private fun buildSchema(): String {
        val json = lastJson ?: return ""
        val segs = json.getJSONArray("segments")
        val arr = JSONArray()
        for (i in 0 until segs.length()) {
            val s = segs.getJSONObject(i)
            arr.put(
                JSONObject()
                    .put("speaker_id", s.getString("speaker"))
                    .put("start_ms", (s.getDouble("start") * 1000).toLong())
                    .put("end_ms", (s.getDouble("end") * 1000).toLong())
                    .put("text", s.getString("text"))
            )
        }
        return JSONObject().put("transcript", arr).toString(2)
    }

    // ---------- Клики ----------
    private fun setupClicks() {
        recordButton.setOnClickListener { toggleRecord() }
        fileButton.setOnClickListener { pickFile() }
        runButton.setOnClickListener { runPipeline() }
        playButton.setOnClickListener { togglePlay() }
        tabText.setOnClickListener { showTab("text") }
        tabSummary.setOnClickListener { showTab("summary") }
        tabChapters.setOnClickListener { showTab("chapters") }
        exportButton.setOnClickListener { exportAll() }
        backButton.setOnClickListener {
            showScreen("record")
            updateSourceLabel()
        }
        seekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser) {
                    mediaPlayer?.seekTo(progress)
                    timeText.text = fmtTime(progress.toLong())
                }
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })
    }
}
