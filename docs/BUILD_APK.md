# Как собрать APK приложения Meevo

Приложение — Kotlin-проект в `android/`. Собирается через Gradle + Android Gradle
Plugin (AGP). Всё, что относится к моделям и нативным библиотекам, уже
подготовлено проектом, но сами веса моделей в git не лежат (`.gitignore`) — их
нужно скачать и положить в assets.

## 0. Итоговая картина

| Что | Статус |
|---|---|
| Исходники Kotlin (`MainActivity`, `Pipeline`, `PunctEngine`, `LlmEngine`, …) | ✅ в `android/` |
| Нативные библиотеки (`sherpa-onnx` .aar, `onnxruntime` .aar, `onnxruntime-genai` .aar) | ✅ в `android/libs/` |
| Модели ASR / диаризации / пунктуации / LLM | ❌ надо скачать |
| JDK 17 | ❌ (сейчас Java 25) |
| Android SDK platform 34 + build-tools | ❌ (есть только android-37 + build-tools 36) |
| Gradle (wrapper) | ❌ (wrapper в проекте отсутствует) |

Требования сборки: AGP **8.5.2**, Kotlin **1.9.24**, `compileSdk 34`, `minSdk 26`,
`targetSdk 34`, `abiFilters arm64-v8a` (APK пойдёт только на arm64-устройствах),
Java **17**.

---

## 1. Окружение

### 1.1. JDK 17
Установи любой JDK 17 (например Temurin):

```powershell
winget install EclipseAdoptium.Temurin.17.JDK
```

Убедись, что `java -version` показывает 17 (иначе задай `JAVA_HOME` вручную,
потому что в системе сейчас стоит Java 25, несовместимая с AGP 8.5).

### 1.2. Android SDK
SDK уже есть в `C:\Users\Daniil\AppData\Local\Android\Sdk`, но там только
`platforms/android-37` и `build-tools/36.0.0`, а нужен **android-34**.

1. Установи `cmdline-tools` (если ещё нет) — проще всего через Android Studio,
   либо скачай zip с https://developer.android.com/studio#command-line-tools-only
   и распакуй в `C:\Users\Daniil\AppData\Local\Android\Sdk\cmdline-tools\latest`.
2. Прими лицензии и доставь нужные пакеты:

```powershell
$sdk = "$env:LOCALAPPDATA\Android\Sdk"
& "$sdk\cmdline-tools\latest\bin\sdkmanager.bat" `
  "platforms;android-34" "build-tools;34.0.0" "platform-tools"
& "$sdk\cmdline-tools\latest\bin\sdkmanager.bat" --licenses
```

3. Создай `android/local.properties` (он в `.gitignore`):

```
sdk.dir=C\:\\Users\\Daniil\\AppData\\Local\\Android\\Sdk
```

### 1.3. Gradle
Wrapper'а в проекте нет. Два варианта:

```powershell
# Вариант A: поставить Gradle и собрать напрямую
winget install Gradle.Gradle
cd E:\nuclear_hack_meevo\android
gradle wrapper --gradle-version 8.9   # один раз, создаст gradlew
.\gradlew.bat assembleDebug
```

Либо сразу `gradle assembleDebug` (Gradle сам подхватит AGP 8.5.2).
Рекомендуемая версия Gradle для AGP 8.5 — **8.7–8.9**.

---

## 2. Модели в assets

Приложение на первом запуске копирует `assets/models/*` во внешнее хранилище и
загружает их. Требуемая структура:

```
android/app/src/main/assets/models/
├── asr/                        # GigaAM v3 (nemo_transducer, 64 mel-бина)
│   ├── encoder.int8.onnx
│   ├── decoder.onnx
│   ├── joiner.onnx
│   └── tokens.txt
├── diar/
│   ├── model.fixed.onnx        # pyannote segmentation-3.0 (топологически отсортированный)
│   └── 3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
├── punct/
│   ├── rupunct_small_int8.onnx
│   └── tokenizer.json
└── llm/                        # НЕобязательно (см. §2.4)
    ├── model.onnx
    ├── model.onnx.data
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── genai_config.json
    └── chat_template.jinja
```

Плюс (рекомендуется для первого запуска) демо-аудио:

```
android/app/src/main/assets/audio/demo_long.wav   # любой WAV 16 кГц моно,
                                                  # например data/test/ruls_0001.wav
```

> Важно: в приложении ASR — это **GigaAM v3** (nemo_transducer), а не
> zipformer, который используется в Python-пайплайне `tools/`. Это разные модели.

### 2.1. ASR — GigaAM v3 (~221 МБ)
Источник: репозиторий `DaniilYemelyanenka/rashifruy` (файлы через Git LFS /
`media.githubusercontent.com`). SHA-256 и URL — в `models/gigaam_v3/MANIFEST.md`.

```powershell
$dir = "E:\nuclear_hack_meevo\android\app\src\main\assets\models\asr"
New-Item -ItemType Directory -Force $dir | Out-Null
$base = "https://media.githubusercontent.com/media/DaniilYemelyanenka/rashifruy/main/src/mobile/android/app/src/main/assets/gigaam"
curl.exe -L -o "$dir\encoder.int8.onnx" "$base/encoder.int8.onnx"
curl.exe -L -o "$dir\decoder.onnx"      "$base/decoder.onnx"
curl.exe -L -o "$dir\joiner.onnx"       "$base/joiner.onnx"
curl.exe -L -o "$dir\tokens.txt" "https://raw.githubusercontent.com/DaniilYemelyanenka/rashifruy/main/src/mobile/android/app/src/main/assets/gigaam/tokens.txt"
```

### 2.2. Диаризация — pyannote + CAM++
CAM++ уже скачан в `models/diar/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx` —
просто скопируй его в `assets/models/diar/`.

Для pyannote нужен **топологически отсортированный** `model.fixed.onnx`: на
Android ORT 1.23 исходный `model.onnx` падает с
`Nodes in a graph must be topologically sorted`. Сгенерируй его из уже скачанного
`models/diar/sherpa-onnx-pyannote-segmentation-3-0/model.onnx`:

```python
# topo_sort.py
import onnx
from onnx import helper

def topo_sort(model):
    g = model.graph
    nodes = list(g.node)
    available = {i.name for i in g.input} | {x.name for x in g.initializer}
    ordered, pending = [], nodes[:]
    while pending:
        progressed = False
        for n in list(pending):
            if all(inp in available for inp in n.input):
                ordered.append(n)
                pending.remove(n)
                available |= set(n.output)
                progressed = True
        if not progressed:
            raise RuntimeError("cycle or unresolved input")
    new_g = helper.make_graph(ordered, g.name, g.input, g.output,
                              g.initializer, g.value_info, g.doc_string)
    new_g.sparse_initializer.extend(g.sparse_initializer)
    m = helper.make_model(new_g, opset_imports=model.opset_import,
                          ir_version=model.ir_version,
                          producer_name=model.producer_name)
    m.metadata_props.extend(model.metadata_props)
    return m

src = "E:/nuclear_hack_meevo/models/diar/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
dst = "E:/nuclear_hack_meevo/android/app/src/main/assets/models/diar/model.fixed.onnx"
onnx.save(topo_sort(onnx.load(src)), dst)
onnx.checker.check_model(onnx.load(dst))
print("ok:", dst)
```

```powershell
E:\nuclear_hack_meevo\.venv\Scripts\python.exe -m pip install onnx
E:\nuclear_hack_meevo\.venv\Scripts\python.exe topo_sort.py
```

### 2.3. Пунктуация — RUPunct-small (~29 МБ)
Источник: HF `ekhodzitsky/rupunct-small-onnx` (MIT). Нужны 2 файла:
`rupunct_small_int8.onnx` и `tokenizer.json`.

```powershell
$dir = "E:\nuclear_hack_meevo\android\app\src\main\assets\models\punct"
New-Item -ItemType Directory -Force $dir | Out-Null
$base = "https://huggingface.co/ekhodzitsky/rupunct-small-onnx/resolve/c4a35be7e4a413cdfb08427db75b621b571569c9"
curl.exe -L -o "$dir\rupunct_small_int8.onnx" "$base/rupunct_small_int8.onnx"
curl.exe -L -o "$dir\tokenizer.json"        "$base/tokenizer.json"
```

> Без `punct/` приложение не стартует: `prepareModels()` безусловно создаёт
> `PunctEngine`, и при отсутствии модели падает в catch → «Ошибка подготовки».

### 2.4. LLM (опционально, ~1.1 ГБ)
LLM нужен только для «LLM-анализ» (конспект/главы/склейка реплик). Если папки
`llm/` нет — этот шаг просто пропустится (обёрнут в try/catch), ASR+диаризация+
пунктуация работают без него.

Готового ONNX-артефакта у модели нет; int4-комплект собирается локально через
`onnxruntime-genai.models.builder` (см. `models/llm/MANIFEST.md` и
`models/llm/export_qvikhr_int4.py`). Модель: `Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning`
(Apache-2.0, BF16 ~3.44 ГБ → int4 ~1.1 ГБ). Это отдельная и самая тяжёлая часть.

---

## 3. Сборка

```powershell
cd E:\nuclear_hack_meevo\android
.\gradlew.bat assembleDebug
```

Выходной APK:

```
android/app/build/outputs/apk/debug/app-debug.apk
```

- `assembleDebug` даёт debug-подписанный APK — его можно сразу установить
  (`adb install app-debug.apk`).
- `assembleRelease` соберёт unsigned-APK (без подписи его не установить; нужна
  собственная keystore-конфигурация).

Проверка установки (устройство по USB / эмулятор):

```powershell
adb install -r android\app\build\outputs\apk\debug\app-debug.apk
```

---

## 4. Что проверено / известные ограничения

- Пайплайн (те же модели, sherpa-onnx 2.0.1) на ПК воспроизводит baseline:
  ASR small-zipformer WER 0.146881, диаризация DER 0.480088, слияние 13/13 JSON.
- На устройстве (отчёт `reports/profiler/DEVICE_REPORT.md`): ASR и диаризация
  работают (ASR нужно чанковать на >20 с), LLM на MiniCPM5 через ORT GenAI
  0.19.0 — не замерен (проблема токенизатора). В приложении LLM заменён на
  QVikhr, и его запуск на устройстве отдельно не подтверждён.
- `abiFilters arm64-v8a` — приложение не запустится на 32-битных/эмуляторах x86.
