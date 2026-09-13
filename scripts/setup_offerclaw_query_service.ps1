[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$Python = "D:\anaconda3\envs\pytorch\python.exe",
    [string]$RepoRoot = "",
    [string]$TaskName = "OfferClaw Query Service",
    [int]$Port = 8000,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
if ($Port -ne 8000) { throw "Only the canonical OfferClaw port 8000 is supported" }
if (-not $RepoRoot) { $RepoRoot = Split-Path -Parent $PSScriptRoot }
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$Python = (Resolve-Path -LiteralPath $Python).Path
$runnerPath = (Resolve-Path -LiteralPath (Join-Path $RepoRoot "scripts\run_offerclaw_query_service.py")).Path
$runtimeRoot = Join-Path $env:USERPROFILE ".offerclaw-runtime"
$tokenPath = Join-Path $runtimeRoot "wechat-query.token"
$listeners = @(Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
$allProcesses = @(Get-CimInstance Win32_Process)
$matching = @($allProcesses | Where-Object {
    ([string]$_.ExecutablePath) -eq $Python -and (
        ([string]$_.CommandLine) -match [regex]::Escape($runnerPath) -or (
            ([string]$_.CommandLine) -match "rag_api:app" -and
            ([string]$_.CommandLine) -match "--port\s+8000(?:\s|$)"
        )
    )
})
$orphanReloadChildren = @($allProcesses | Where-Object {
    $listenerOwners = @($listeners | ForEach-Object { [int]$_.OwningProcess })
    $listenerOwners -contains [int]$_.ParentProcessId -and
    ([string]$_.ExecutablePath) -eq $Python -and
    ([string]$_.CommandLine) -match "multiprocessing\.spawn"
})
if ($listeners.Count -gt 0 -and $matching.Count -eq 0 -and $orphanReloadChildren.Count -eq 0) {
    throw "Port 8000 has a listener but no exact OfferClaw rag_api process; refusing replacement"
}

$arguments = '"' + $runnerPath + '"'
if ($DryRun) {
    [pscustomobject]@{
        status = "dry_run"; task = $TaskName; port = $Port; workers = 1; reload = $false
    } | ConvertTo-Json -Compress
    exit 0
}

New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
if (-not (Test-Path -LiteralPath $tokenPath -PathType Leaf)) {
    $bytes = New-Object byte[] 32
    $generator = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $generator.GetBytes($bytes) } finally { $generator.Dispose() }
    $token = ([BitConverter]::ToString($bytes) -replace "-", "").ToLowerInvariant()
    [IO.File]::WriteAllText($tokenPath, $token, [Text.Encoding]::ASCII)
}
& icacls.exe $runtimeRoot /inheritance:r /grant:r "${env:USERNAME}:(OI)(CI)F" "SYSTEM:(OI)(CI)F" | Out-Null
& icacls.exe $tokenPath /inheritance:r /grant:r "${env:USERNAME}:F" "SYSTEM:F" | Out-Null
$dpapiPath = Join-Path $runtimeRoot "openai-key.dpapi"
if (Test-Path -LiteralPath $dpapiPath -PathType Leaf) {
    & icacls.exe $dpapiPath /inheritance:r /grant:r "${env:USERNAME}:F" "SYSTEM:F" | Out-Null
}

if ($PSCmdlet.ShouldProcess($TaskName, "register single-worker OfferClaw query service")) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    $stopIds = [Collections.Generic.HashSet[int]]::new()
    foreach ($process in $matching) { [void]$stopIds.Add([int]$process.ProcessId) }
    foreach ($process in $orphanReloadChildren) { [void]$stopIds.Add([int]$process.ProcessId) }
    $changed = $true
    while ($changed) {
        $changed = $false
        foreach ($process in $allProcesses) {
            if ($stopIds.Contains([int]$process.ParentProcessId) -and -not $stopIds.Contains([int]$process.ProcessId)) {
                [void]$stopIds.Add([int]$process.ProcessId)
                $changed = $true
            }
        }
    }
    foreach ($processId in @($stopIds) | Sort-Object -Descending) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 500
    if (Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        throw "The exact OfferClaw port 8000 process did not stop cleanly"
    }
    $action = New-ScheduledTaskAction -Execute $Python -Argument $arguments -WorkingDirectory $RepoRoot
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
    Start-ScheduledTask -TaskName $TaskName
}

$token = [IO.File]::ReadAllText($tokenPath, [Text.Encoding]::ASCII).Trim()
$headers = @{ "X-OfferClaw-Internal-Token" = $token; "X-OfferClaw-Traffic-Origin" = "wechat_direct" }
$health = $null
for ($attempt = 0; $attempt -lt 120; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/internal/wechat-health" -Headers $headers -TimeoutSec 2
        if ($health.status -eq "ok") { break }
    } catch {
        Start-Sleep -Milliseconds 500
    }
}
if ($null -eq $health -or $health.status -ne "ok") {
    throw "OfferClaw query service failed its authenticated health check"
}
[pscustomobject]@{
    status = "ok"; task = $TaskName; port = $Port; workers = 1; reload = $false
    query_service_version = $health.query_service_version
    repository_fingerprint = $health.repository_fingerprint
    runtime_status = $health.runtime.status
} | ConvertTo-Json -Compress
