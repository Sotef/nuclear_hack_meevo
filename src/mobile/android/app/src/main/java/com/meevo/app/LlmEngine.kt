package com.meevo.app

import ai.onnxruntime.genai.Generator
import ai.onnxruntime.genai.GeneratorParams
import ai.onnxruntime.genai.Model
import ai.onnxruntime.genai.Sequences
import ai.onnxruntime.genai.Tokenizer

class LlmEngine(modelDir: String, private val systemPrompt: String) : AutoCloseable {
    private val model = Model(modelDir)
    private val tokenizer = Tokenizer(model)

    private fun generate(system: String, user: String, maxNewTokens: Int): String {
        // Qwen3-ChatML ровно как tools/run_llm/run_llm.py (postproc_v1):
        // applyChatTemplate в ORT GenAI 0.19.0 падает (SIGABRT Scudo — баг JNI),
        // поэтому рендерим шаблон вручную. Пробел перед " /no_think" обязателен.
        val prompt = buildString {
            append("<|im_start|>system\n")
            append(system.trim())
            append("<|im_end|>\n")
            append("<|im_start|>user\n")
            append(user)
            append(" /no_think<|im_end|>\n")
            append("<|im_start|>assistant\n")
        }
        val seqs: Sequences = tokenizer.encode(prompt)
        val promptLen = seqs.getSequence(0).size
        GeneratorParams(model).use { params ->
            params.setSearchOption("max_length", (promptLen + maxNewTokens).toDouble())
            params.setSearchOption("do_sample", false)
            Generator(model, params).use { gen ->
                gen.appendTokenSequences(seqs)
                while (!gen.isDone) gen.generateNextToken()
                val out = gen.getSequence(0)
                val genOnly = if (out.size > promptLen) out.copyOfRange(promptLen, out.size) else out
                var text = tokenizer.decode(genOnly).trim()
                text = text.replace(Regex("(?s)^\\s*<think>.*?</think>\\s*"), "")
                for (marker in listOf("/no_think", "no_think", "/think")) {
                    if (text.startsWith(marker)) {
                        text = text.removePrefix(marker).trimStart()
                        break
                    }
                }
                text = text
                    .replace(Regex("(?s)<\\|im_start\\|>.*"), "")
                    .trim()
                seqs.close()
                return text
            }
        }
    }

    fun punctuate(raw: String, maxNewTokens: Int = 256): String {
        val input = raw.trim()
        if (input.isEmpty()) return raw
        val text = generate(systemPrompt, "Текст: $input", maxNewTokens)
            .replace(Regex("(?s)^\\s*Исправленный текст:\\s*"), "")
            .trim()
        val norm: (String) -> String = { it.lowercase().replace('ё', 'е') }
        val inTokens = norm(input).split(Regex("[^а-яa-z0-9]+")).filter { it.isNotBlank() }
        val outTokens = norm(text).split(Regex("[^а-яa-z0-9]+")).filter { it.isNotBlank() }
        val inWords = inTokens.filter { it.length >= 4 }
        val kept = inWords.count { iw ->
            outTokens.any { ow ->
                ow == iw ||
                    (ow.length >= 4 && ow.startsWith(iw.take(5))) ||
                    (ow.length >= 4 && iw.startsWith(ow.take(5)))
            }
        }
        val keepRatio = if (inWords.isEmpty()) 1.0 else kept.toDouble() / inWords.size
        val echo = text.contains("Верни только") || text.contains("Объяснение:") ||
            text.contains("Хорошо, я") || text.contains("исправленный текст")
        val growsTooMuch = outTokens.size > inTokens.size + 2
        val degenerate = text.isBlank() || echo || growsTooMuch || keepRatio < 0.8
        android.util.Log.d(
            "Meevo",
            "llm guard: in=${inTokens.size}/${inWords.size} out=${outTokens.size} kept=$kept ratio=" +
                String.format("%.2f", keepRatio) + " grow=$growsTooMuch deg=$degenerate"
        )
        if (degenerate) return input
        return text
    }

    /** Диалоговый ввод: каждая фраза на своей строке, формат "MM:SS SPEAKER: текст". */
    fun summarize(dialog: String): String {
        val system = "Ты составляешь краткий конспект разговора на русском. " +
            "НЕ копируй реплики дословно и НЕ указывай таймкоды/время — только содержание. " +
            "Объединяй смежные темы в один пункт, не повторяй одну и ту же мысль. " +
            "3-7 пунктов маркированным списком (каждый пункт с новой строки, начинай с «•»). " +
            "Не добавляй фактов, которых нет в диалоге. Только список пунктов."
        return generate(system, "Диалог:\n$dialog", 512)
    }

    /** Возвращает JSON-массив глав: [{"start":"MM:SS","end":"MM:SS","title":"..."}]. */
    fun chapters(dialog: String): String {
        val system = "Ты делишь разговор на тематические блоки (главы), как таймкоды на видео. " +
            "Верни СТРОГО JSON-массив без пояснений и без markdown: " +
            "[{\"start\": \"MM:SS\", \"end\": \"MM:SS\", \"title\": \"Краткий заголовок\"}]. " +
            "Границы блоков бери из времени фраз диалога. 2-8 блоков."
        return generate(system, "Диалог:\n$dialog", 512)
    }

    /**
     * Ищет фразы, случайно разрезанные на две реплики (например, одна фраза
     * разделена между двумя говорящими из-за ошибки диаризации).
     * Возвращает JSON-массив пар номеров строк: [{"first": N, "second": M, "reason": "..."}].
     */
    fun findMerges(dialog: String): String {
        val system = "Ты проверяешь транскрипцию разговора. " +
            "Иногда одна фраза случайно разрезана на две реплики: предложение оборвано между сегментами " +
            "или одна фраза разделена между двумя говорящими из-за ошибки диаризации. " +
            "Найди такие пары СОСЕДНИХ строк, которые по смыслу составляют ОДНУ фразу. " +
            "Верни СТРОГО JSON-массив без markdown: [{\"first\": N, \"second\": M, \"reason\": \"кратко\"}] — " +
            "номера строк диалога с 1. Не склеивай разные предложения, вопрос-ответ, смену темы. " +
            "Если всё в порядке — верни []."
        return generate(system, "Диалог:\n$dialog", 256)
    }

    override fun close() {
        tokenizer.close()
        model.close()
    }
}
