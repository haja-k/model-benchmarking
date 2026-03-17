# ==============================================================
# LLM Timeout Diagnostic Runner - All Phases, All Models
# ==============================================================
# Runs timeout_test.py (Phases 1-5) for each configured model.
# HTML reports saved to results/timeout_test/
#
# Usage:
#   .\run_timeout_test.ps1
#   .\run_timeout_test.ps1 -Model "si-qwen3-vl-30b"
#   .\run_timeout_test.ps1 -Single
#   .\run_timeout_test.ps1 -SloTtft 5000
# ==============================================================

param(
    [string] $Model        = "",
    [switch] $Single,
    [switch] $Agentic,
    [switch] $IncludeAgentic,
    [int]    $PromptTokens = 1000,
    [int]    $MaxTokens    = 512,
    [int]    $MinTokens    = 256,
    [float]  $SloTtft      = 3000,
    [float]  $SloTpot      = 150,
    [float]  $SloTtlt      = 60000,
    [float]  $StallMs      = 5000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ALL_MODELS = @(
    "deepseek-ai/DeepSeek-V3.2",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "openai/gpt-oss-120b"
)

$targets = if ($Model -ne "") { @($Model) } else { $ALL_MODELS }

$commonArgs = @(
    "--prompt-tokens", $PromptTokens,
    "--max-tokens",    $MaxTokens,
    "--min-tokens",    $MinTokens,
    "--slo-ttft",      $SloTtft,
    "--slo-tpot",      $SloTpot,
    "--slo-ttlt",      $SloTtlt,
    "--stall-ms",      $StallMs
)
if ($Single)         { $commonArgs += "--single" }
if ($Agentic)        { $commonArgs += "--agentic" }
if ($IncludeAgentic) { $commonArgs += "--include-agentic" }

$results    = @()
$totalStart = Get-Date
$sepEq      = "=" * 62
$sepDash    = "-" * 62

Write-Host ""
Write-Host $sepEq
Write-Host "  LLM TIMEOUT DIAGNOSTIC RUNNER"
Write-Host $sepEq
Write-Host "  Models : $($targets -join ', ')"
if ($Single) {
    Write-Host "  Phases : Smoke test (TC-03 only)"
} elseif ($Agentic) {
    Write-Host "  Phases : Phase 6 - Agentic workload only (TC-A1..TC-A4)"
} elseif ($IncludeAgentic) {
    Write-Host "  Phases : 1-5 (full suite) + Phase 6 (agentic)"
} else {
    Write-Host "  Phases : 1-5 (full suite)"
}
Write-Host "  SLOs   : TTFT=$($SloTtft)ms  TPOT=$($SloTpot)ms  TTLT=$($SloTtlt)ms"
Write-Host "  Stall  : >$($StallMs)ms chunk gap"
Write-Host "  Output : results/timeout_test/"
Write-Host $sepEq

foreach ($m in $targets) {
    $modelStart = Get-Date
    Write-Host ""
    Write-Host $sepDash
    Write-Host "  Model: $m"
    Write-Host $sepDash

    $exitCode = 0
    try {
        & python timeout_test.py --model $m @commonArgs
        $exitCode = $LASTEXITCODE
    }
    catch {
        Write-Host "  [ERROR] Failed to run timeout_test.py for '$m': $_" -ForegroundColor Red
        $exitCode = 1
    }

    $elapsed    = (Get-Date) - $modelStart
    $elapsedStr = "{0:mm\:ss}" -f $elapsed
    $status     = if ($exitCode -eq 0) { "OK" } else { "FAILED (exit $exitCode)" }
    $color      = if ($exitCode -eq 0) { "Green" } else { "Red" }

    $results += [PSCustomObject]@{
        Model   = $m
        Status  = $status
        Elapsed = $elapsedStr
    }

    Write-Host ""
    Write-Host "  Finished $m - $status in $elapsedStr" -ForegroundColor $color
}

$totalElapsed    = (Get-Date) - $totalStart
$totalElapsedStr = "{0:mm\:ss}" -f $totalElapsed

Write-Host ""
Write-Host $sepEq
Write-Host "  RUNNER SUMMARY"
Write-Host $sepEq
$results | Format-Table -AutoSize | Out-String | Write-Host
Write-Host "  Total wall time : $totalElapsedStr"

$reportPath = Join-Path $PSScriptRoot "results\timeout_test"
if (Test-Path $reportPath) {
    Write-Host "  Reports saved to: $(Resolve-Path $reportPath)"
} else {
    Write-Host "  Reports saved to: $reportPath"
}

Write-Host $sepEq
Write-Host ""
