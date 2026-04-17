# GCPC

**GCPC** is a Windows desktop application for controlling a PC with hand gestures through a webcam.

The project uses **Python**, **MediaPipe**, **OpenCV**, and **PySide6** to recognize gestures in real time and map them to desktop actions such as hotkeys, gesture sequences, and mouse control.

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
- friendly popup window on critical errors

### Distribution

- build into a single `exe`
- packaged together with:
  - `config.json`
  - runtime logs directory
  - task model for the backend

---

## Tech stack

- Python
- PySide6
- OpenCV
- MediaPipe

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
4. Make sure your hands are visible in the camera frame.
5. Open `Gesture settings` if you want to change gesture bindings, calibration options, or mouse-area parameters.

---

## Default setup

The current default configuration is based on `GCPC/config.json`.

### Hand roles

By default:

- `DOMINANT = RIGHT`
- `SUPPORT = LEFT`

This means the application expects the **right hand** to be the dominant hand and the **left hand** to be the non-dominant or support hand.

### Video defaults

By default:

- camera index: `0`
- resolution: `640x360`
- mirror mode: `true`
- FPS counter: `enabled`

---

## Supported gestures

GCPC currently includes the following core gestures:

- `OPEN_PALM`
- `FIST`
- `PINCH`
- `PINCH_MIDDLE`
- `THUMBS_UP`
- `SWIPE_LEFT`
- `SWIPE_RIGHT`

These gestures can be used either as standalone actions or as part of a sequence.

---

## Default control modes

GCPC supports several runtime modes. With the current default config, they are switched by special functional gestures.

### Enter mouse mode

- `NON_DOMINANT-THUMBS_UP`

With the default hand roles, this means:

- **left hand thumbs up** -> enter `MOUSE` mode

### Enter one-hand mode

- `BOTH-THUMBS_UP`

This enables `ONE-HAND` mode.

### Enter sequence recording mode

- `BOTH-FIST`

This enables `RECORD` mode for gesture-sequence input.

### Exit the current mode

- `BOTH-OPEN_PALM`

By default, exit is triggered after a short hold.

---

## How mouse control works

GCPC supports a dedicated mouse mode.

### Default mouse flow

1. Enable `Camera`.
2. Enable `Hand control`.
3. Show **thumbs up with the non-dominant hand** to enter `MOUSE` mode.
4. Move the **dominant hand** in front of the camera.
5. Use the configured pointer landmark area to move the cursor.
6. Use click gestures with the other hand.
7. Exit mouse mode with `BOTH-OPEN_PALM`.

### Default cursor movement

By default, cursor movement uses:

- `pointer_hand = DOMINANT`
- `pointer_landmark = 8`

Landmark `8` is the **index fingertip**, so the cursor is controlled by the **dominant hand index finger**.

### Mouse control rectangle

The cursor is mapped inside a normalized control rectangle:

- `x = 0.45`
- `y = 0.3`
- `width = 0.5`
- `height = 0.5`

When mouse mode is active, this area is shown as a **yellow rectangle** in the camera window.

If the cursor feels too compressed, too sensitive, or does not cover the full screen comfortably, adjust the mouse area in `Gesture settings`.

### Default mouse buttons

With the current default config:

- `left click = NON_DOMINANT-PINCH`
- `right click = NON_DOMINANT-PINCH_MIDDLE`

With default hand roles, this means:

- **left hand pinch** -> left mouse button
- **left hand thumb + middle pinch** -> right mouse button

### Default scrolling

Scrolling is enabled by default and uses:

- `hand = RIGHT`
- `gesture = FIST`
- `landmark = 0`

This means:

- make a **right-hand fist**
- move the hand vertically
- the application converts wrist movement into mouse wheel scrolling

If scrolling feels too fast or too sensitive, change the scroll settings in `config.json` or through the settings UI if exposed there.

### Practical mouse usage example

With the default config:

1. Show **thumbs up with the left hand** to enter mouse mode.
2. Move the **right index finger** inside the yellow control rectangle to move the cursor.
3. Do **left-hand pinch** for left click.
4. Do **left-hand pinch middle** for right click.
5. Make a **right-hand fist** and move vertically to scroll.
6. Show **both open palms** to exit mouse mode.

---

## Gesture configuration

GCPC allows you to bind recognized hand gestures to desktop actions.

There are two main binding types:

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

This mode is safer for important or global commands because it reduces false activations.

---

## How to configure gestures in settings

When creating gesture bindings, it is recommended to follow these rules.

### Use simple and familiar hotkeys

Prefer standard desktop shortcuts such as:

- `Ctrl+C`
- `Ctrl+V`
- `Ctrl+Z`
- `Alt+Tab`
- `Win+D`
- `Ctrl+Shift+Esc`

### Reserve single gestures for safe actions

Single gestures are better for actions that are not dangerous if triggered accidentally.

Good examples:

- show desktop
- confirm
- cancel
- media control
- simple navigation shortcuts

### Reserve sequences for sensitive actions

Use gesture sequences for commands that should not be triggered by mistake.

Good examples:

- task manager
- app switching
- desktop automation shortcuts
- multi-key system commands

### Keep mappings easy to remember

A gesture should feel related to the action whenever possible.

Examples:

- `OPEN_PALM` -> show desktop
- `FIST` -> escape or cancel
- `THUMBS_UP` -> confirm or accept

### Avoid overly long sequences

Long sequences are harder to perform consistently and may reduce usability.

A practical sequence length is usually:

- 2 gestures for common protected actions
- 3 gestures for rare or system-level actions

> **Note**
> The exact text format for hotkeys depends on the current parser used by the application.
> In most cases, bindings are expected in a standard hotkey form such as `Ctrl+C`, `Alt+Tab`, or `Win+D`.
> If your build uses a different syntax, update the examples above to match the actual implementation.

---

## Default command mappings

The current config also contains default desktop actions.

### Functional mappings

- `BOTH-FIST` -> `MODE_RECORD`
- `NON_DOMINANT-THUMBS_UP` -> `MODE_MOUSE`
- `BOTH-THUMBS_UP` -> `MODE_ONE_HAND`
- `BOTH-OPEN_PALM` -> `MODE_EXIT`

### Single gesture mappings

- `NON_DOMINANT-PINCH` -> `CTRL+C`
- `NON_DOMINANT-FIST` -> `CTRL+V`
- `DOMINANT-SWIPE_LEFT` -> `ALT+SHIFT+TAB`
- `DOMINANT-SWIPE_RIGHT` -> `ALT+TAB`

### Complex gesture mappings

- `DOMINANT-FIST > DOMINANT-FIST` -> `CTRL+SHIFT+N`

> **Important**
> The exact runtime behavior always follows `config.json`.
> If you change bindings in settings, that file becomes the source of truth.

---

## Sequence mode basics

Sequence mode allows building a gesture sequence and then committing it as an action.

### Default sequence flow

1. Enter sequence mode with `BOTH-FIST`.
2. Perform gestures with the configured input hand.
3. Confirm candidate gestures with the confirm binding.
4. Commit the sequence with the commit binding.

### Default sequence-related bindings

- input hand: `DOMINANT`
- confirm: `NON_DOMINANT-PINCH`
- undo: `NON_DOMINANT-OPEN_PALM > NON_DOMINANT-FIST`
- commit: `BOTH-FIST`

This mode is useful for protected commands and multi-step shortcuts.

---

## Calibration

Calibration can be started from the settings window using:

`Run calibration now`

You can also start calibration by pressing:

- `C`

Calibration helps the application adapt gesture recognition to the current user, camera position, and lighting conditions.

Recommended cases for recalibration:

- first launch
- camera angle changed
- lighting changed
- recognition quality became unstable

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
- mouse control
- calibration workflow
- packaged Windows distribution

Some gestures and runtime improvements are still in progress.

---

## Planned improvements

- extended gesture set
- better sequence editor
- improved mouse-control tuning
- clearer settings UX
- more robust runtime diagnostics
