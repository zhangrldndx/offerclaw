[CmdletBinding()]
param(
    [string]$Python = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    [string]$InstallRoot = "$env:USERPROFILE\.offerclaw-runtime\wechat-bridge-venv"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$requirements = Join-Path $repoRoot "requirements-wechat-bridge.txt"
$venvPython = Join-Path $InstallRoot "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python 3.12 not found: $Python"
}
if (-not (Test-Path -LiteralPath $requirements -PathType Leaf)) {
    throw "Bridge requirements not found: $requirements"
}
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    & $Python -m venv $InstallRoot
}
& $venvPython -m pip install --disable-pip-version-check --progress-bar off -r $requirements
& $venvPython -c "import pydantic, requests; print('Windows bridge runtime OK')"
Write-Output $venvPython
