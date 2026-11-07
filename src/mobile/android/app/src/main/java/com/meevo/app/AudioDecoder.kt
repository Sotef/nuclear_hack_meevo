package com.meevo.app

import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.util.Log
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Декодирует произвольный аудиофайл (wav/mp3/ogg/flac/m4a) в моно float[] 16 кГц.
 * Порт AudioDecoder из репозитория rashifruy (MediaExtractor + MediaCodec),
 * даунмикс и линейный ресемпл — здесь.
 */
object AudioDecoder {
    private const val TAG = "Meevo"
    private const val DEQUEUE_TIMEOUT_US = 10_000L
    private const val MAX_IDLE_LOOPS = 200

    fun decodeToMono16k(file: File, targetSampleRate: Int = 16000): FloatArray {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(file.absolutePath)
            var trackIndex = -1
            for (i in 0 until extractor.trackCount) {
                val mime = extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME)
                if (mime != null && mime.startsWith("audio/")) {
                    trackIndex = i
                    break
                }
            }
            if (trackIndex < 0) throw IOException("В файле нет аудиодорожки")
            val format = extractor.getTrackFormat(trackIndex)
            extractor.selectTrack(trackIndex)

            val mime = format.getString(MediaFormat.KEY_MIME)
                ?: throw IOException("Неизвестный mime у аудиодорожки")
            var sourceRate = if (format.containsKey(MediaFormat.KEY_SAMPLE_RATE))
                format.getInteger(MediaFormat.KEY_SAMPLE_RATE) else targetSampleRate
            var channels = if (format.containsKey(MediaFormat.KEY_CHANNEL_COUNT))
                format.getInteger(MediaFormat.KEY_CHANNEL_COUNT) else 1
            var encoding = if (format.containsKey(MediaFormat.KEY_PCM_ENCODING))
                format.getInteger(MediaFormat.KEY_PCM_ENCODING) else android.media.AudioFormat.ENCODING_PCM_16BIT

            val out = FloatList()
            if (MediaFormat.MIMETYPE_AUDIO_RAW == mime) {
                val buffer = ByteBuffer.allocateDirect(1 shl 20).order(ByteOrder.LITTLE_ENDIAN)
                while (true) {
                    buffer.clear()
                    val size = extractor.readSampleData(buffer, 0)
                    if (size < 0) break
                    buffer.position(0)
                    buffer.limit(size)
                    appendPcm(buffer, encoding, channels, out)
                    extractor.advance()
                }
            } else {
                val codec = MediaCodec.createDecoderByType(mime)
                try {
                    codec.configure(format, null, null, 0)
                    codec.start()
                    var inputBuffers = codec.inputBuffers
                    var outputBuffers = codec.outputBuffers
                    val info = MediaCodec.BufferInfo()
                    var inputDone = false
                    var outputDone = false
                    var idle = 0
                    while (!outputDone) {
                        var progress = false
                        if (!inputDone) {
                            val inIndex = codec.dequeueInputBuffer(DEQUEUE_TIMEOUT_US)
                            if (inIndex >= 0) {
                                val inBuffer = inputBuffers[inIndex]
                                inBuffer.clear()
                                val sampleSize = extractor.readSampleData(inBuffer, 0)
                                if (sampleSize < 0) {
                                    codec.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                    inputDone = true
                                } else {
                                    codec.queueInputBuffer(inIndex, 0, sampleSize, extractor.sampleTime, 0)
                                    extractor.advance()
                                }
                                progress = true
                            }
                        }
                        val outIndex = codec.dequeueOutputBuffer(info, DEQUEUE_TIMEOUT_US)
                        if (outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                            val of = codec.outputFormat
                            if (of.containsKey(MediaFormat.KEY_SAMPLE_RATE)) sourceRate = of.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                            if (of.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) channels = of.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                            if (of.containsKey(MediaFormat.KEY_PCM_ENCODING)) encoding = of.getInteger(MediaFormat.KEY_PCM_ENCODING)
                            progress = true
                        } else if (outIndex == MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED) {
                            outputBuffers = codec.outputBuffers
                            progress = true
                        } else if (outIndex >= 0) {
                            val outBuffer = outputBuffers[outIndex]
                            outBuffer.position(info.offset)
                            outBuffer.limit(info.offset + info.size)
                            appendPcm(outBuffer, encoding, channels, out)
                            codec.releaseOutputBuffer(outIndex, false)
                            progress = true
                            if ((info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) outputDone = true
                        }
                        if (!progress) {
                            idle++
                            if (idle > MAX_IDLE_LOOPS) {
                                Log.w(TAG, "decoder idle, stop")
                                break
                            }
                        } else idle = 0
                    }
                } finally {
                    try {
                        codec.stop()
                    } catch (_: Exception) {
                    }
                    codec.release()
                }
            }
            if (out.size() == 0) return FloatArray(0)
            return resampleLinear(out.toArray(), sourceRate, targetSampleRate)
        } finally {
            extractor.release()
        }
    }

    private fun appendPcm(buffer: ByteBuffer, encoding: Int, channelsIn: Int, out: FloatList) {
        val channels = if (channelsIn <= 0) 1 else channelsIn
        when (encoding) {
            android.media.AudioFormat.ENCODING_PCM_FLOAT -> {
                val frames = buffer.remaining() / (4 * channels)
                for (i in 0 until frames) {
                    var sum = 0f
                    for (c in 0 until channels) sum += buffer.float
                    out.add(sum / channels)
                }
            }
            android.media.AudioFormat.ENCODING_PCM_8BIT -> {
                val frames = buffer.remaining() / channels
                for (i in 0 until frames) {
                    var sum = 0
                    for (c in 0 until channels) sum += (buffer.get().toInt() and 0xFF) - 128
                    out.add((sum / channels.toFloat()) / 128f)
                }
            }
            else -> {
                val frames = buffer.remaining() / (2 * channels)
                for (i in 0 until frames) {
                    var sum = 0
                    for (c in 0 until channels) sum += buffer.short.toInt()
                    out.add((sum / channels.toFloat()) / 32768f)
                }
            }
        }
    }

    private fun resampleLinear(input: FloatArray, sourceRate: Int, targetRate: Int): FloatArray {
        if (sourceRate <= 0 || targetRate <= 0 || sourceRate == targetRate || input.isEmpty()) return input
        val outLen = maxOf(1, Math.round(input.size.toDouble() * targetRate / sourceRate).toInt())
        val out = FloatArray(outLen)
        val step = sourceRate.toDouble() / targetRate
        for (i in 0 until outLen) {
            val pos = i * step
            val idx = pos.toInt()
            val next = minOf(idx + 1, input.size - 1)
            val frac = (pos - idx).toFloat()
            out[i] = input[idx] * (1f - frac) + input[next] * frac
        }
        return out
    }

    private class FloatList {
        private var data = FloatArray(16000 * 30)
        private var size = 0

        fun add(v: Float) {
            if (size == data.size) {
                val grown = FloatArray(data.size + (data.size shr 1))
                System.arraycopy(data, 0, grown, 0, size)
                data = grown
            }
            data[size++] = v
        }

        fun size(): Int = size

        fun toArray(): FloatArray {
            val r = FloatArray(size)
            System.arraycopy(data, 0, r, 0, size)
            return r
        }
    }
}
