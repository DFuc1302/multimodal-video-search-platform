$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir

$python = Join-Path $projectDir ".venv_submission_windows\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Dang tao moi truong Windows nhe cho app nop bai..."
    py -3.11 -m venv (Join-Path $projectDir ".venv_submission_windows")
    & $python -m pip install --disable-pip-version-check -r (Join-Path $projectDir "requirements_submission_windows.txt")
}

$credentialFile = Join-Path $projectDir ".dres_credentials.env"
if (-not (Test-Path $credentialFile)) {
    throw "Khong tim thay file cau hinh DRES: $credentialFile"
}
foreach ($line in Get-Content -LiteralPath $credentialFile) {
    $trimmed = $line.Trim()
    if (-not $trimmed -or $trimmed.StartsWith("#")) { continue }
    $parts = $trimmed.Split("=", 2)
    if ($parts.Count -ne 2) { throw "Dong cau hinh khong hop le trong .dres_credentials.env" }
    Set-Item -Path ("Env:" + $parts[0].Trim()) -Value $parts[1]
}

$env:DATA_ROOT = if ($env:DATA_ROOT) { $env:DATA_ROOT } else { Join-Path (Split-Path -Parent $projectDir) "final_search_index" }
$env:MAPFILES_DIR = if ($env:MAPFILES_DIR) { $env:MAPFILES_DIR } else { Join-Path $env:DATA_ROOT "map-keyframes" }
$env:DRES_OFFICIAL_MAPS = if ($env:DRES_OFFICIAL_MAPS) { $env:DRES_OFFICIAL_MAPS } else { Join-Path $env:DATA_ROOT "aic26-b2-map-keyframes.zip" }
$port = if ($env:SERVER_PORT) { $env:SERVER_PORT } else { "8091" }

Write-Host "App nop bai dang chay tai http://localhost:$port"
Write-Host "Nhan Ctrl+C de dung."
& $python -m uvicorn submission_server:app --host 127.0.0.1 --port $port --no-access-log
