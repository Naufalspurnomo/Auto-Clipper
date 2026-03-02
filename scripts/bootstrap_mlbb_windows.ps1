Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "[bootstrap-windows] $Message" -ForegroundColor Cyan
}

function Invoke-CmdArray {
    param(
        [string[]]$Command,
        [string[]]$Arguments
    )
    $exe = $Command[0]
    $prefix = @()
    if ($Command.Count -gt 1) {
        $prefix = $Command[1..($Command.Count - 1)]
    }
    & $exe @prefix @Arguments
}

function Resolve-PythonCommand {
    $candidates = @(
        @("python"),
        @("py", "-3")
    )

    foreach ($candidate in $candidates) {
        try {
            $versionText = Invoke-CmdArray -Command $candidate -Arguments @(
                "-c",
                "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')"
            )
            if ($LASTEXITCODE -eq 0 -and $versionText) {
                return @{
                    Command = $candidate
                    Version = [version]$versionText.Trim()
                }
            }
        } catch {
            continue
        }
    }

    throw "Python not found. Install Python 3.10+ first from https://www.python.org/downloads/windows/"
}

function Assert-Tool {
    param(
        [string]$Name,
        [string]$InstallHint,
        [string]$PathHint
    )

    if (Get-Command $Name -ErrorAction SilentlyContinue) {
        Write-Step "$Name found in PATH."
    } else {
        Write-Warning "$Name not found in PATH."
        Write-Host $InstallHint
        Write-Host $PathHint
    }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot
Write-Step "Repo root: $repoRoot"

$pythonInfo = Resolve-PythonCommand
$pythonVersion = $pythonInfo.Version
if ($pythonVersion -lt [version]"3.10.0") {
    throw "Python 3.10+ required. Current detected version: $pythonVersion"
}
Write-Step "Python version OK: $pythonVersion"

$venvPath = Join-Path $repoRoot ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Step "Creating virtual environment at .venv"
    Invoke-CmdArray -Command $pythonInfo.Command -Arguments @("-m", "venv", ".venv")
} else {
    Write-Step "Using existing virtual environment at .venv"
}

$activateScript = Join-Path $venvPath "Scripts/Activate.ps1"
if (-not (Test-Path $activateScript)) {
    throw "Venv activation script not found: $activateScript"
}

Write-Step "Activating virtual environment"
. $activateScript

$venvPython = Join-Path $venvPath "Scripts/python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Python executable in venv not found: $venvPython"
}

Write-Step "Installing Python dependencies from requirements.txt"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

Assert-Tool -Name "ffmpeg" `
    -InstallHint @"
Install FFmpeg:
  choco install ffmpeg
  winget install --id Gyan.FFmpeg -e
"@ `
    -PathHint @"
Example PATH setup:
  Current session:
    `$env:Path += ';C:\ffmpeg\bin'
  Permanent (new terminal required):
    setx PATH "`$env:PATH;C:\ffmpeg\bin"
"@

Assert-Tool -Name "tesseract" `
    -InstallHint @"
Install Tesseract OCR:
  choco install tesseract
  winget install --id UB-Mannheim.TesseractOCR -e
"@ `
    -PathHint @"
Example PATH setup:
  Current session:
    `$env:Path += ';C:\Program Files\Tesseract-OCR'
  Permanent (new terminal required):
    setx PATH "`$env:PATH;C:\Program Files\Tesseract-OCR"
"@

if (Get-Command "yt-dlp" -ErrorAction SilentlyContinue) {
    Write-Step "yt-dlp found in PATH."
} else {
    Write-Warning "yt-dlp not found in PATH. Installing into venv with pip."
    & $venvPython -m pip install yt-dlp
}

Write-Step "Running smoke test: python main.py --help"
& $venvPython main.py --help | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Smoke test failed: main.py --help"
}

Write-Step "Bootstrap completed successfully."
Write-Host ""
Write-Host "Next commands:"
Write-Host "  python main.py --url `"https://www.youtube.com/watch?v=VIDEO_ID`" --platform tiktok --dry-run"
