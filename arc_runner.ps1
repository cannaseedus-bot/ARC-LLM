param(
    [string]$Command = "info",
    [string]$ModelPath = "arc_model_shards",
    [string]$Prompt = "Once upon a time"
)

function Show-ARCLogo {
    Write-Host "ARC-LLM Unified" -ForegroundColor Cyan
}

function Get-ShardInfo {
    param([string]$Path)
    if (!(Test-Path "$Path/manifest.json")) {
        Write-Host "Manifest not found at $Path" -ForegroundColor Yellow
        return
    }
    $manifest = Get-Content "$Path/manifest.json" | ConvertFrom-Json
    Write-Host "Shards: $($manifest.num_shards)" -ForegroundColor Green
}

Show-ARCLogo
switch($Command) {
    "info" { Get-ShardInfo -Path $ModelPath }
    default { Write-Host "Unknown command: $Command" }
}
