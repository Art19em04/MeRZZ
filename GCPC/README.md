# GCPC

GCPC — приложение для управления ПК жестами рук через камеру (Windows, Python + MediaPipe + OpenCV + PySide6).

## Что умеет сейчас

- Панель управления при старте:
  - `Hand control: ON/OFF`
  - `Camera: ON/OFF`
  - выбор разрешения камеры в dropdown
  - кнопка `Gesture settings`
- Окно настроек жестов:
  - соответствия `gesture -> hotkey` (single и sequence)
  - запуск калибровки кнопкой `Run calibration now`
  - настройка размера mouse-области
  - чекбокс `Display only hands windows` (режим "только руки")
- Логи и friendly-error обработка:
  - лог-файл `logs/gcpc.log`
  - всплывающее окно при критической ошибке
- Сборка в один `exe` (+ `config.json` + модель для tasks backend)

## Быстрый старт (из исходников)

```powershell
pip install -r requirements.txt
python -m app.main
```

## Управление в рантайме

1. Запустите приложение.
2. В панели включите `Camera`.
3. Включите `Hand control`.
4. Откройте `Gesture settings`, если нужно поменять биндинги.

Горячая клавиша открытия настроек: `Ctrl+,`

## Backend детектора: Legacy и Tasks

Трекер автоматически выбирает backend:

1. `mediapipe.solutions` (legacy), если доступен.
2. `mediapipe.tasks` (HandLandmarker), если legacy недоступен.

Это сделано, чтобы приложение одинаково работало в IDE и в `exe` на новых версиях `mediapipe` (например `0.10.33`).

## Лево/право руки (важно)

В проекте включена нормализация handedness (label + геометрия + учет `mirror`), чтобы уменьшить расхождения между legacy/tasks.

При необходимости можно явно задать стратегию в `config.json`:

```json
{
  "detector": {
    "handedness": {
      "strategy": "auto",
      "swap_labels": false,
      "prefer_geometry_on_conflict": true
    }
  }
}
```

`strategy`:
- `auto` — по умолчанию
- `label` — доверять label от модели
- `geometry` — определять сторону по геометрии руки

## Метрики и eval

Метрики/замеры выключены по умолчанию.

Чтобы включить:

```json
{
  "measurements": {
    "enabled": true
  }
}
```

## Сборка EXE (onefile)

См. подробности в [BUILD.md](./BUILD.md).

Коротко:

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1 -InstallMissingDeps
```

Или с явным Python:

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1 -PythonExe "C:\Users\Artyom\AppData\Local\Programs\Python\Python312\python.exe" -InstallMissingDeps
```

После сборки:

- `release\GCPC.exe`
- `release\config.json`
- `release\logs\gcpc.log`
- `release\models\hand_landmarker.task`

## Галерея жестов

Положите изображения в папку:

- `docs/gestures/`

Рекомендуемые имена файлов:

- `open_palm.png`
- `fist.png`
- `pinch.png`
- `pinch_middle.png`
- `thumbs_up.png`
- `swipe_left.png`
- `swipe_right.png`

Пример красивой вставки в README:

```markdown
## Галерея жестов

| OPEN_PALM | FIST | PINCH |
|---|---|---|
| ![OPEN_PALM](docs/gestures/open_palm.png) | ![FIST](docs/gestures/fist.png) | ![PINCH](docs/gestures/pinch.png) |

| PINCH_MIDDLE | THUMBS_UP | SWIPE_LEFT / SWIPE_RIGHT |
|---|---|---|
| ![PINCH_MIDDLE](docs/gestures/pinch_middle.png) | ![THUMBS_UP](docs/gestures/thumbs_up.png) | ![SWIPE](docs/gestures/swipe_combo.png) |
```

Или крупными карточками:

```markdown
![OPEN_PALM](docs/gestures/open_palm.png)
![FIST](docs/gestures/fist.png)
![PINCH](docs/gestures/pinch.png)
```

## Где смотреть логи

- `logs/gcpc.log` (при запуске из исходников)
- `release/logs/gcpc.log` (после сборки `exe`)

Если приложение падает, присылайте последний traceback из этого файла.
