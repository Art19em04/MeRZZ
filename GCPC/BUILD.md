# GCPC Build Guide (Windows)

## 1) Install runtime dependencies

```powershell
pip install -r requirements.txt
```

## 2) Install build dependency

```powershell
pip install -r requirements-build.txt
```

## 3) Build one-file executable

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

If `PyInstaller` is missing, you can let script install build deps automatically:

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1 -InstallMissingDeps
```

This mode also installs runtime dependencies from `requirements.txt` if modules like `cv2` are missing.

## Build result

After success, you get:

- `release\GCPC.exe`
- `release\config.json`
- `release\logs\` (log folder for runtime errors)
- `release\models\hand_landmarker.task`

## Notes

- The executable is built in `--onefile` mode with bundled `mediapipe`, `opencv-python`, and `PySide6`.
- Build script ensures `models\hand_landmarker.task` exists (downloads it automatically if missing).
- App logs are written to `logs\gcpc.log` near the executable working directory (for source runs: project `logs\gcpc.log`).
- Gesture settings can be changed in-app via:
  - control panel button: `Gesture settings`
  - shortcut: `Ctrl+,`
