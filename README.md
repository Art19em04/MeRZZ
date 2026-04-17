# GCPC

**GCPC** is a Windows desktop application for controlling a PC with hand gestures through a webcam.

The project uses **Python**, **MediaPipe**, **OpenCV**, and **PySide6** to recognize hand gestures in real time and map them to desktop actions such as hotkeys, gesture sequences, and mouse control.

---

## Overview

GCPC is designed as a practical gesture-control utility for Windows. It provides a desktop UI for enabling camera tracking, turning gesture control on or off, configuring gesture bindings, calibrating recognition, and packaging the application into a standalone executable.

At the current stage, the project already includes a working control panel, gesture settings, calibration запуск, logging, critical-error handling, and one-file EXE build support.

---

## Features

### Main control panel

At startup, the application provides a control panel with:

- `Hand control: ON/OFF`
- `Camera: ON/OFF`
- camera resolution selection via dropdown
- `Gesture settings` button

### Gesture settings window

The settings window currently supports:

- mapping `gesture -> hotkey`
- mapping both **single gestures** and **gesture sequences**
- launching calibration with `Run calibration now`
- configuring mouse control area size
- `Display only hands windows` mode

### Logging and error handling

- log file: `logs/gcpc.log`
- user-friendly popup window on critical errors

### Distribution

- build into a single `exe`
- packaged together with:
  - `config.json`
  - runtime logs directory
  - task model for the backend

---

## Tech stack

- **Python**
- **PySide6**
- **OpenCV**
- **MediaPipe**

---

## Quick start

### Run from source

```powershell
pip install -r requirements.txt
python -m app.main
```

---

## Runtime usage

1. Launch the application.
2. Turn on `Camera`.
3. Turn on `Hand control`.
4. Open `Gesture settings` if you want to change gesture bindings, calibration options, or mouse-area parameters.

---

## Gesture configuration

GCPC allows you to bind recognized hand gestures to desktop actions.

There are two main binding types.

### 1. Single gesture

A single recognized gesture triggers one action.

Examples:

- `OPEN_PALM` -> `Win+D`
- `FIST` -> `Esc`
- `THUMBS_UP` -> `Enter`

This mode is faster, but may be more sensitive to accidental activation.

### 2. Gesture sequence

A sequence requires several gestures in a specific order before an action is triggered.

Examples:

- `OPEN_PALM -> PINCH` -> `Ctrl+C`
- `PINCH -> THUMBS_UP` -> `Alt+Tab`
- `FIST -> OPEN_PALM -> PINCH` -> `Ctrl+Shift+Esc`

This mode is safer for important commands because it reduces false activations.

---

## How to configure commands in settings

When creating gesture bindings, it is recommended to follow these rules.

### Use simple and familiar hotkeys

Prefer standard desktop shortcuts such as:

- `Ctrl+C`
- `Ctrl+V`
- `Ctrl+Z`
- `Alt+Tab`
- `Win+D`
- `Ctrl+Shift+Esc`

### Use single gestures for safe actions

Single gestures are better for actions that are not dangerous if triggered accidentally.

Good examples:

- show desktop
- confirm
- cancel
- media control
- simple navigation shortcuts

### Use sequences for sensitive actions

Use gesture sequences for commands that should not be triggered by mistake.

Good examples:

- task manager
- app switching
- system shortcuts
- automation shortcuts

### Keep mappings intuitive

A gesture should feel related to the action whenever possible.

Examples:

- `OPEN_PALM` -> show desktop
- `FIST` -> escape or cancel
- `THUMBS_UP` -> confirm or accept

### Avoid very long sequences

Long sequences are harder to perform consistently and reduce usability.

A practical sequence length is usually:

- **2 gestures** for common protected actions
- **3 gestures** for rare or system-level actions

> **Note**
>
> The exact text format for hotkeys depends on the current parser used by the application.
> In most cases, bindings are expected in a standard hotkey form such as `Ctrl+C`, `Alt+Tab`, or `Win+D`.
> If your build uses a different syntax, update the examples above to match the actual implementation.

---

## Recommended binding examples

### Basic desktop control

- `OPEN_PALM` -> `Win+D`
- `FIST` -> `Esc`
- `THUMBS_UP` -> `Enter`
- `PINCH` -> `Space`

### Navigation and multitasking

- `PINCH -> THUMBS_UP` -> `Alt+Tab`
- `OPEN_PALM -> OPEN_PALM` -> `Win+Tab`
- `FIST -> PINCH` -> `Ctrl+W`

### Productivity shortcuts

- `OPEN_PALM -> PINCH` -> `Ctrl+C`
- `PINCH -> OPEN_PALM` -> `Ctrl+V`
- `FIST -> THUMBS_UP` -> `Ctrl+Z`

These are only examples. The best mapping depends on your use case and how stable each gesture is in your environment.

---

## Calibration

Calibration can be started from the settings window using:

`Run calibration now`

Calibration helps the application adapt gesture recognition to the current user, camera position, and lighting conditions.

Recalibration is recommended when:

- launching the app for the first time
- the camera angle has changed
- lighting conditions have changed
- gesture recognition became unstable

---

## Mouse area configuration

The settings window allows configuring the size of the mouse-control area.

This is useful for balancing:

- precision
- comfort
- cursor reach
- sensitivity

A smaller area may feel more precise.
A larger area may feel more natural but can require wider hand movement.

---

## Display only hands windows

`Display only hands windows` enables a hand-focused display mode.

This mode is useful for:

- debugging gesture recognition
- checking whether hands are detected correctly
- testing camera framing and lighting
- reducing visual clutter during setup

---

## Build EXE

Detailed build instructions are available in [BUILD.md](GCPC/BUILD.md).

### One-file build

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1 -InstallMissingDeps
```

### Build output

After the build, the following files will be available:

- `release\GCPC.exe`
- `release\config.json`
- `release\logs\gcpc.log`
- `release\models\hand_landmarker.task`

---

## Logs

Use these log files for troubleshooting:

- `GCPC/logs/gcpc.log` when running from source
- `GCPC/release/logs/gcpc.log` after building the executable

---

## Gesture gallery

| OPEN_PALM | FIST | PINCH |
|---|---|---|
| ![OPEN_PALM](GCPC/docs/gestures/open_palm.png) | ![FIST](GCPC/docs/gestures/fist.png) | ![PINCH](GCPC/docs/gestures/pinch.png) |

| PINCH_MIDDLE | THUMBS_UP | SWIPE_LEFT / SWIPE_RIGHT |
|---|---|---|
| ![PINCH_MIDDLE](GCPC/docs/gestures/pinch_middle.png) | ![THUMBS_UP](GCPC/docs/gestures/thumbs_up.png) | IN_PROGRESS |

---

## Project structure

```text
GCPC/
├─ app/
├─ docs/
│  └─ gestures/
├─ logs/
├─ models/
├─ release/
├─ build.ps1
├─ requirements.txt
├─ config.json
└─ BUILD.md
```

---

## Current status

GCPC currently focuses on:

- real-time hand gesture recognition
- gesture-to-hotkey mapping
- gesture sequences
- calibration workflow
- packaged Windows distribution

Some gestures and runtime improvements are still in progress.

---

## Future improvements

- expanded gesture set
- improved sequence editor
- better mouse-control tuning
- clearer settings UX
- more robust runtime diagnostics
