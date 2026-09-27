param(
    [string]$PythonPath = 'python',
    [int]$SampleSize = 400,
    [int]$RunLimit = 400,
    [string]$RunName = 'pilot400',
    [string]$DatasetCache = ''
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if ($SampleSize -lt 1 -or $RunLimit -lt 1 -or $RunLimit -gt $SampleSize) {
    throw 'Require 1 <= RunLimit <= SampleSize.'
}
if ($RunName -notmatch '^[A-Za-z0-9_-]+$') {
    throw 'RunName must use letters, digits, underscores or hyphens.'
}
if ($DatasetCache) { $env:HF_DATASETS_CACHE = $DatasetCache }
$env:PYTHONUTF8 = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$ModelDirectory = Join-Path $ProjectRoot 'models/qwen3-vl-4b'
$DataDirectory = Join-Path $ProjectRoot "data/gqa_dev$SampleSize"
$RunDirectory = Join-Path $ProjectRoot "runs/$RunName"
$ReportDirectory = Join-Path $ProjectRoot "reports/$RunName"
$GalleryDirectory = Join-Path $ProjectRoot "local-gallery/$RunName"

function Invoke-PythonStep([string[]]$PythonArguments) {
    & $PythonPath @PythonArguments
    if ($LASTEXITCODE -ne 0) { throw "Python step failed with exit code $LASTEXITCODE" }
}

Invoke-PythonStep @('-m','lookagain.cli','fetch-model','--output',$ModelDirectory,'--revision','ebb281ec70b05090aa6165b016eac8ec08e71b17')
Invoke-PythonStep @('-m','lookagain.data','--output',$DataDirectory,'--limit',"$SampleSize",'--revision','a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8')
Invoke-PythonStep @('-m','lookagain.cli','run','--manifest',"$DataDirectory/manifest.jsonl",'--model-dir',$ModelDirectory,'--config',"$ProjectRoot/configs/pilot.json",'--output',$RunDirectory,'--limit',"$RunLimit")
Invoke-PythonStep @('-m','lookagain.cli','analyze','--run',$RunDirectory,'--output',$ReportDirectory)
Invoke-PythonStep @("$ProjectRoot/scripts/diagnose_pilot.py",'--run',$RunDirectory,'--output',$ReportDirectory)
Invoke-PythonStep @("$ProjectRoot/scripts/describe_slices.py",'--run',$RunDirectory,'--manifest',"$DataDirectory/manifest.jsonl",'--output',$ReportDirectory)
Invoke-PythonStep @('-m','lookagain.cli','visualize','--run',$RunDirectory,'--manifest',"$DataDirectory/manifest.jsonl",'--summary',"$ReportDirectory/summary.json",'--output',$GalleryDirectory)
Write-Output "Completed paired report: $ReportDirectory/report.md"
Write-Output "Local image gallery: $GalleryDirectory/gallery.html"
