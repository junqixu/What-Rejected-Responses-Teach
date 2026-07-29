param(
    [string]$Python = "python",
    [int]$ValidationCalls = 600,
    [int]$TestCalls = 1500
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

$DeterministicTypes = "dependency_redirect,extra_edge,missing_node,spurious_node,disorder,wrong_target"
$DeepSeekTypes = "computation,operation_substitution"

& $Python -m rebuttal.generation.deterministic_backfill `
    --input data/rebuttal_v2/source/validation_clean.jsonl `
    --out data/rebuttal_v3/deepseek/validation `
    --split validation `
    --error_types $DeterministicTypes
if ($LASTEXITCODE -ne 0) { throw "validation deterministic backfill failed" }

& $Python -m rebuttal.generation.deepseek_eight_types `
    --input data/rebuttal_v2/source/validation_clean.jsonl `
    --out data/rebuttal_v3/deepseek/validation `
    --split validation `
    --error_types $DeepSeekTypes `
    --max_api_calls $ValidationCalls `
    --max_retries 3 `
    --sleep_seconds 0.5
if ($LASTEXITCODE -ne 0) { throw "validation DeepSeek generation failed" }

& $Python -m rebuttal.generation.deterministic_backfill `
    --input data/rebuttal_v2/source/test_clean.jsonl `
    --out data/rebuttal_v3/deepseek/test `
    --split test `
    --error_types $DeterministicTypes
if ($LASTEXITCODE -ne 0) { throw "test deterministic backfill failed" }

& $Python -m rebuttal.generation.deepseek_eight_types `
    --input data/rebuttal_v2/source/test_clean.jsonl `
    --out data/rebuttal_v3/deepseek/test `
    --split test `
    --error_types $DeepSeekTypes `
    --max_api_calls $TestCalls `
    --max_retries 3 `
    --sleep_seconds 0.5
if ($LASTEXITCODE -ne 0) { throw "test DeepSeek generation failed" }

& $Python -m rebuttal.generation.package_release `
    --dataset_root data/rebuttal_v3/deepseek `
    --source_root data/rebuttal_v2/source `
    --out data/rebuttal_release `
    --overwrite true
if ($LASTEXITCODE -ne 0) { throw "release packaging failed" }
