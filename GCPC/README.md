# Microgestures MVP (Chest-cam, pinch clutch)

## What this is
Real-time pipeline to trigger system shortcuts (Copy/Paste/Cut/Undo) via **microgestures** (thumb swipe/tap along index) with a **pinch clutch** window to avoid "Midas touch". Camera: chest-level, 60+ FPS. Backend: ONNX Runtime (DirectML) hand landmarker + custom microgesture detector + OS key events.

## Quick start
1. Python 3.10+ recommended.
2. `pip install -r requirements.txt`
3. Put ONNX models to `models/`:
   - `hand_landmarker.onnx`
   - `palm_detection.onnx`
4. Windows only: run a terminal with normal user rights. We emit key events through `ctypes` `SendInput`.
5. Run:
   ```
   python -m app.main
   ```
6. Keys in the preview window:
   - `ESC` — quit
   - `f` — mark "false positive" for the last action (telemetry)
   - `SPACE` — toggle HUD

## Config
See `config.json`. You can tune `roi`, `conf_min`, `min_duration_ms`, `cooldown_ms`, `clutch.window_ms`, etc.

## KPI tracking
- End-to-end latency p95 (frame in → OS event) <= 100 ms
- Macro accuracy > 95% on user tasks
- False activations < 1/hour

## Notes
- For chest-cam, keep the hand in the ROI rectangle; illumination changes may harm tracking.
- We log everything in `logs/session_*.jsonl`. Aggregate later to compute p95, FP/hour, confusion counts.
