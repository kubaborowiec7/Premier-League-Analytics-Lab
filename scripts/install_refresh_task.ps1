param(
    [string]$TaskName = "Premier League Analytics Refresh",
    [string]$DailyTime = "07:15"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$RefreshScript = Join-Path $ProjectRoot "scripts\refresh_current_data.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project virtual environment is missing: $Python"
}

$Action = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument ('"' + $RefreshScript + '"') `
    -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Daily -At $DailyTime
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2)
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Refreshes immutable football snapshots and current dashboard artifacts." `
    -Force

Write-Output "Registered '$TaskName' at $DailyTime; missed runs start when Windows is available."
