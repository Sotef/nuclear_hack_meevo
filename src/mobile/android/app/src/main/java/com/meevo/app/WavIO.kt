package com.meevo.app

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

object WavIO {
    data class Audio(val samples: FloatArray, val sampleRate: Int)

    fun read(file: File): Audio {
        val bytes = file.readBytes()
        require(bytes.size > 44) { "too small wav" }
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        require(bb.getInt(0) == 0x46464952) { "not RIFF" }
        var pos = 12
        var channels = 1
        var sampleRate = 16000
        var bits = 16
        var audioFormat = 1
        var dataOffset = -1
        var dataLen = 0
        while (pos + 8 <= bytes.size) {
            val id = String(bytes, pos, 4, Charsets.US_ASCII)
            val size = bb.getInt(pos + 4)
            if (id == "fmt ") {
                audioFormat = bb.getShort(pos + 8).toInt()
                channels = bb.getShort(pos + 10).toInt()
                sampleRate = bb.getInt(pos + 12)
                bits = bb.getShort(pos + 22).toInt()
            } else if (id == "data") {
                dataOffset = pos + 8
                dataLen = minOf(size, bytes.size - dataOffset)
                break
            }
            pos += 8 + size + (size and 1)
        }
        require(dataOffset >= 0) { "no data chunk" }
        val bytesPer = channels * (bits / 8)
        val frames = dataLen / bytesPer
        val out = FloatArray(frames)
        for (i in 0 until frames) {
            var acc = 0f
            for (c in 0 until channels) {
                val off = dataOffset + (i * channels + c) * (bits / 8)
                acc += when {
                    audioFormat == 3 && bits == 32 -> bb.getFloat(off)
                    bits == 16 -> bb.getShort(off) / 32768f
                    bits == 32 -> bb.getInt(off) / 2147483648f
                    bits == 8 -> (bb.get(off).toInt() - 128) / 128f
                    else -> 0f
                }
            }
            out[i] = acc / channels
        }
        return Audio(resample(out, sampleRate, 16000), 16000)
    }

    private fun resample(x: FloatArray, from: Int, to: Int): FloatArray {
        if (from == to || x.isEmpty()) return x
        val n = (x.size.toLong() * to / from).toInt()
        val y = FloatArray(n)
        for (i in 0 until n) {
            val t = i.toDouble() * from / to
            val i0 = t.toInt().coerceIn(0, x.size - 1)
            val i1 = (i0 + 1).coerceAtMost(x.size - 1)
            val f = (t - i0).toFloat()
            y[i] = x[i0] * (1 - f) + x[i1] * f
        }
        return y
    }

    fun write(file: File, samples: FloatArray, sampleRate: Int) {
        val dataLen = samples.size * 2
        val bb = ByteBuffer.allocate(44 + dataLen).order(ByteOrder.LITTLE_ENDIAN)
        bb.put("RIFF".toByteArray()); bb.putInt(36 + dataLen); bb.put("WAVE".toByteArray())
        bb.put("fmt ".toByteArray()); bb.putInt(16); bb.putShort(1); bb.putShort(1)
        bb.putInt(sampleRate); bb.putInt(sampleRate * 2); bb.putShort(2); bb.putShort(16)
        bb.put("data".toByteArray()); bb.putInt(dataLen)
        for (s in samples) bb.putShort((s.coerceIn(-1f, 1f) * 32767).toInt().toShort())
        file.writeBytes(bb.array())
    }
}
