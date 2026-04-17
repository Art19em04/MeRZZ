param(
    [string]$PythonExe = "python",
    [string]$AppName = "GCPC",
    [switch]$InstallMissingDeps
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$pythonCmd = Get-Command $PythonExe -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    $pythonCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $PythonExe = "py"
    }
}
if (-not $pythonCmd) {
    throw "Python executable not found. Install Python 3.10+ and ensure it is in PATH."
}

$runtimeModules = @("cv2", "mediapipe", "PySide6", "numpy")

function Test-PythonModule {
    param(
        [string]$Python,
        [string]$ModuleName
    )
    $code = "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$ModuleName') else 1)"
    & $Python -c $code *> $null
    return ($LASTEXITCODE -eq 0)
}

$distDir = Join-Path $root "dist"
$buildDir = Join-Path $root "build\pyinstaller"
$releaseDir = Join-Path $root "release"
$modelsDir = Join-Path $root "models"
$modelPath = Join-Path $modelsDir "hand_landmarker.task"
$modelUrl = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

Write-Host "[build] Root: $root"
Write-Host "[build] Checking PyInstaller..."
$hasPyInstaller = Test-PythonModule -Python $PythonExe -ModuleName "PyInstaller"

if (-not $hasPyInstaller -and $InstallMissingDeps) {
    Write-Host "[build] PyInstaller not found. Installing build dependencies..."
    & $PythonExe -m pip install -r (Join-Path $root "requirements-build.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install build dependencies. Run '$PythonExe -m pip install -r requirements-build.txt' manually."
    }
    $hasPyInstaller = Test-PythonModule -Python $PythonExe -ModuleName "PyInstaller"
}

if (-not $hasPyInstaller) {
    throw "PyInstaller is not installed. Run '$PythonExe -m pip install -r requirements-build.txt' or restart with -InstallMissingDeps."
}

$missingRuntime = @()
foreach ($module in $runtimeModules) {
    if (-not (Test-PythonModule -Python $PythonExe -ModuleName $module)) {
        $missingRuntime += $module
    }
}

if ($missingRuntime.Count -gt 0 -and $InstallMissingDeps) {
    Write-Host "[build] Missing runtime modules: $($missingRuntime -join ', ')"
    Write-Host "[build] Installing runtime dependencies..."
    & $PythonExe -m pip install -r (Join-Path $root "requirements.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install runtime dependencies. Run '$PythonExe -m pip install -r requirements.txt' manually."
    }
    $missingRuntime = @()
    foreach ($module in $runtimeModules) {
        if (-not (Test-PythonModule -Python $PythonExe -ModuleName $module)) {
            $missingRuntime += $module
        }
    }
}

if ($missingRuntime.Count -gt 0) {
    throw "Missing runtime modules in this Python environment: $($missingRuntime -join ', '). Install requirements.txt and rebuild."
}

if (-not (Test-Path $modelPath)) {
    Write-Host "[build] Hand landmarker model not found at $modelPath"
    Write-Host "[build] Downloading model from MediaPipe storage..."
    if (-not (Test-Path $modelsDir)) {
        New-Item -ItemType Directory -Path $modelsDir | Out-Null
    }
    try {
        Invoke-WebRequest -Uri $modelUrl -OutFile $modelPath -UseBasicParsing
    } catch {
        throw "Could not download hand_landmarker.task. Download manually to '$modelPath' and rerun build."
    }
}

if (-not (Test-Path $modelPath)) {
    throw "Missing task model file: $modelPath"
}

if (Test-Path $releaseDir) {
    Remove-Item -LiteralPath $releaseDir -Recurse -Force
}
if (Test-Path $distDir) {
    Remove-Item -LiteralPath $distDir -Recurse -Force
}
if (Test-Path $buildDir) {
    Remove-Item -LiteralPath $buildDir -Recurse -Force
}
New-Item -ItemType Directory -Path $releaseDir | Out-Null

$pyinstallerArgs = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onefile",
    "--windowed",
    "--name", $AppName,
    "--distpath", $distDir,
    "--workpath", $buildDir,
    "--specpath", $buildDir,
    "--collect-all", "mediapipe",
    "--collect-submodules", "mediapipe",
    "--collect-data", "mediapipe",
    "--collect-binaries", "mediapipe",
    "--collect-all", "cv2",
    "--collect-all", "PySide6",
    "--collect-binaries", "cv2",
    "--hidden-import", "cv2",
    "--hidden-import", "cv2.cv2",
    "--hidden-import", "mediapipe",
    "--hidden-import", "mediapipe.tasks",
    "--hidden-import", "mediapipe.tasks.python",
    "--hidden-import", "mediapipe.tasks.python.vision",
    "--hidden-import", "mediapipe.tasks.python.core.base_options",
    "--hidden-import", "mediapipe.tasks.python.vision.core.vision_task_running_mode",
    "--hidden-import", "mediapipe.solutions",
    "--hidden-import", "mediapipe.solutions.hands",
    "--hidden-import", "mediapipe.python.solutions",
    "--hidden-import", "mediapipe.python.solutions.hands",
    "--hidden-import", "PySide6.QtCore",
    "--hidden-import", "PySide6.QtGui",
    "--hidden-import", "PySide6.QtWidgets",
    "--add-data", "$modelPath;models",
    "app/main.py"
)

Write-Host "[build] Building one-file executable..."
& $PythonExe @pyinstallerArgs
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

$exePath = Join-Path $distDir "$AppName.exe"
if (-not (Test-Path $exePath)) {
    throw "Build completed but executable was not found: $exePath"
}

Copy-Item -LiteralPath $exePath -Destination (Join-Path $releaseDir "$AppName.exe") -Force
Copy-Item -LiteralPath (Join-Path $root "config.json") -Destination (Join-Path $releaseDir "config.json") -Force
New-Item -ItemType Directory -Path (Join-Path $releaseDir "logs") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $releaseDir "models") -Force | Out-Null
Copy-Item -LiteralPath $modelPath -Destination (Join-Path $releaseDir "models\hand_landmarker.task") -Force

Write-Host ""
Write-Host "[build] Done."
Write-Host "[build] Release folder: $releaseDir"
Write-Host "[build] Files:"
Write-Host "  - $releaseDir\$AppName.exe"
Write-Host "  - $releaseDir\config.json"
Write-Host "  - $releaseDir\logs\"
Write-Host "  - $releaseDir\models\hand_landmarker.task"
