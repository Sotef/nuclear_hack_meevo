# Meevo — офлайн-расшифровка разговоров

Android-приложение: запись с микрофона или аудиофайл превращается в полную расшифровку
разговора — с таймкодами, определением говорящих, пунктуацией, конспектом и главами.
**Вся обработка выполняется на устройстве, без сети.**

APK (установка: `adb install -r Meevo.apk`): **[Releases → v0.1-meevo](https://github.com/Sotef/nuclear_hack/releases/tag/v0.1-meevo)**

## Возможности

- Запись с микрофона и загрузка файлов (wav/mp3/ogg/flac/m4a)
- Русская речь → текст с таймкодами (GigaAM v3 RNN-T, WER 0.121 на тестовом наборе)
- Диаризация говорящих с глобальными ID на всю запись
- Пунктуация и регистр за миллисекунды (RUPunct, F1 0.838)
- LLM-анализ: конспект, главы с таймкодами, склейка случайно разрезанных фраз
- Один запуск полного пайплайна с живыми прогрессбарами и ETA по этапам
- Экспорт: JSON (контракт v0.2), SRT, TXT, JSON-схема
- Material 3 UI со светлой и тёмной темой

## Структура репозитория

```
artifacts/
  demo/          — материалы демонстрации (APK-ссылка, дизайн-макет, примеры вывода)
  test-results/  — отчёты экспериментов, results.csv, архив отчётов агентов
  data/          — тестовые и eval-наборы (manifest, референсы)
  models/        — манифесты моделей (веса не в git, восстановление по MANIFEST.md)
  device_bundle/ — консольный набор для замеров на устройстве (adb)
config/          — конфигурация пайплайна (example.env, промпты LLM)
docs/            — архитектура, решения, ресерчи, функции, ограничения, тесты
src/
  mobile/android/— исходники приложения Meevo (Kotlin)
  tools/         — python-инструменты eval/обучения (WER, DER, пунктуация, LoRA)
```

## Документация

| Документ | Содержание |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | архитектура пайплайна и приложения |
| [docs/DECISIONS.md](docs/DECISIONS.md) | ключевые технические решения и обоснования |
| [docs/RESEARCH.md](docs/RESEARCH.md) | ресерч моделей, замеры, лицензии |
| [docs/FEATURES.md](docs/FEATURES.md) | функции приложения |
| [docs/SYSTEM_REQUIREMENTS.md](docs/SYSTEM_REQUIREMENTS.md) | рекомендуемые системные требования |
| [docs/TEST_RESULTS.md](docs/TEST_RESULTS.md) | методика и результаты тестирования |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | ограничения решения |
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | контракт данных (JSON v0.2) и API ML-модуля |

## Быстрый старт

Приложение: скачайте APK из [Releases](https://github.com/Sotef/nuclear_hack/releases/tag/v0.1-meevo),
установите (`adb install -r Meevo.apk` или откройте файл на телефоне), нажмите
**«Обработать всё»** — этапы с прогрессом и ETA идут автоматически.

Сборка из исходников (JDK 17, Android SDK 34):

```bash
cd src/mobile/android
gradle assembleDebug        # APK: app/build/outputs/apk/debug/
```

Модели для сборки в assets не входят в git — восстановление по
`artifacts/models/*/MANIFEST.md` (URL, sha256, лицензии).

## Лицензии

Код — MIT. Модели: GigaAM v3 (MIT), RUPunct (MIT), QVikhr-3 (Apache-2.0),
pyannote segmentation-3.0 (MIT), CAM++ (Apache-2.0), sherpa-onnx (Apache-2.0),
ONNX Runtime (MIT). Лицензии в `src/mobile/android/app/src/main/assets/licenses/`.
