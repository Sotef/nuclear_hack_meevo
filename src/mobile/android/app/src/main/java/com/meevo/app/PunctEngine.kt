package com.meevo.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.File

class PunctEngine(modelDir: File) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val vocab: Map<String, Long>
    private val id2label = arrayOf(
        "UPPER_PERIOD", "LOWER_PERIOD", "UPPER_TOTAL_PERIOD",
        "UPPER_COMMA", "LOWER_COMMA", "UPPER_TOTAL_COMMA",
        "UPPER_QUESTION", "LOWER_QUESTION", "UPPER_TOTAL_QUESTION",
        "UPPER_TIRE", "LOWER_TIRE", "UPPER_TOTAL_TIRE",
        "UPPER_VOSKL", "LOWER_VOSKL", "UPPER_TOTAL_VOSKL",
        "UPPER_DVOETOCHIE", "LOWER_DVOETOCHIE", "UPPER_TOTAL_DVOETOCHIE",
        "UPPER_PERIODCOMMA", "LOWER_PERIODCOMMA", "UPPER_TOTAL_PERIODCOMMA",
        "UPPER_DEFIS", "LOWER_DEFIS", "UPPER_TOTAL_DEFIS",
        "UPPER_QUESTIONVOSKL", "LOWER_QUESTIONVOSKL", "UPPER_TOTAL_QUESTIONVOSKL",
        "UPPER_MNOGOTOCHIE", "LOWER_MNOGOTOCHIE", "UPPER_TOTAL_MNOGOTOCHIE",
        "UPPER_O", "LOWER_O", "UPPER_TOTAL_O"
    )

    private val punctMap = mapOf(
        "O" to "", "PERIOD" to ".", "COMMA" to ",", "QUESTION" to "?",
        "TIRE" to "\u2014", "DVOETOCHIE" to ":", "VOSKL" to "!",
        "PERIODCOMMA" to ";", "DEFIS" to "-", "MNOGOTOCHIE" to "...",
        "QUESTIONVOSKL" to "?!"
    )

    init {
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(2)
        session = env.createSession(File(modelDir, "rupunct_small_int8.onnx").absolutePath, opts)
        val tok = JSONObject(File(modelDir, "tokenizer.json").readText(Charsets.UTF_8))
        val v = tok.getJSONObject("model").getJSONObject("vocab")
        val m = HashMap<String, Long>(v.length() * 2)
        for (key in v.keys()) m[key] = v.getLong(key)
        vocab = m
    }

    fun punctuate(text: String): String {
        val words = text.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
        if (words.isEmpty()) return text
        val subTokens = words.map { wordpiece(it) }
        val chunks = ArrayList<MutableList<Int>>()
        var cur = ArrayList<Int>()
        var curLen = 2
        for (i in words.indices) {
            if (curLen + subTokens[i].size > 508 && cur.isNotEmpty()) {
                chunks.add(cur)
                cur = ArrayList()
                curLen = 2
            }
            cur.add(i)
            curLen += subTokens[i].size
        }
        if (cur.isNotEmpty()) chunks.add(cur)

        val out = StringBuilder()
        for (chunk in chunks) {
            val ids = ArrayList<Long>()
            val wordOf = ArrayList<Int>()
            ids.add(2L); wordOf.add(-1)
            for (wi in chunk) {
                for (t in subTokens[wi]) {
                    ids.add(t); wordOf.add(wi)
                }
            }
            ids.add(3L); wordOf.add(-1)

            val t = ids.size
            val input = Array(1) { ids.toLongArray() }
            val mask = Array(1) { LongArray(t) { 1L } }
            val tt = Array(1) { LongArray(t) { 0L } }
            OnnxTensor.createTensor(env, input).use { idsT ->
                OnnxTensor.createTensor(env, mask).use { maskT ->
                    OnnxTensor.createTensor(env, tt).use { ttT ->
                        session.run(
                            mapOf(
                                "input_ids" to idsT,
                                "attention_mask" to maskT,
                                "token_type_ids" to ttT
                            )
                        ).use { res ->
                            @Suppress("UNCHECKED_CAST")
                            val logits = res.get(0).value as Array<Array<FloatArray>>
                            val pred = IntArray(t)
                            for (j in 0 until t) {
                                val row = logits[0][j]
                                var best = 0
                                var bv = row[0]
                                for (c in 1 until row.size) {
                                    if (row[c] > bv) {
                                        bv = row[c]
                                        best = c
                                    }
                                }
                                pred[j] = best
                            }
                            for (wi in chunk) {
                                if (out.isNotEmpty()) out.append(' ')
                                val firstTok = wordOf.indexOf(wi)
                                val label = id2label[pred[firstTok]]
                                out.append(processToken(words[wi], label))
                            }
                        }
                    }
                }
            }
        }
        return out.toString()
    }

    private fun wordpiece(word: String): List<Long> {
        if (word.length > 100) return listOf(1L)
        val out = ArrayList<Long>()
        var start = 0
        while (start < word.length) {
            var end = word.length
            var cur: Long? = null
            while (start < end) {
                val sub = word.substring(start, end)
                val cand = if (start == 0) sub else "##$sub"
                val id = vocab[cand]
                if (id != null) {
                    cur = id
                    break
                }
                end--
            }
            if (cur == null) return listOf(1L)
            out.add(cur)
            start = end
        }
        return out
    }

    private fun processToken(token: String, label: String): String {
        val case: String
        val cls: String
        when {
            label.startsWith("UPPER_TOTAL_") -> {
                case = "upper"; cls = label.removePrefix("UPPER_TOTAL_")
            }
            label.startsWith("UPPER_") -> {
                case = "cap"; cls = label.removePrefix("UPPER_")
            }
            else -> {
                case = "lower"; cls = label.removePrefix("LOWER_")
            }
        }
        val word = when (case) {
            "upper" -> token.uppercase()
            "cap" -> token.replaceFirstChar { it.titlecase() }
            else -> token
        }
        val p = punctMap[cls] ?: ""
        return if (p == "\u2014" && case != "lower") "$word $p" else "$word$p"
    }

    override fun close() {
        session.close()
    }
}
