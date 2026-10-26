param (
    [string]$IQTree2Log = "time_log_iqtree2.tsv",
    [string]$IQTree3Log = "time_log_iqtree3.tsv",
    [string]$Platform   = ""
)

# Compare IQ-TREE 3 peak memory against the IQ-TREE 2 baseline + threshold.
# Rows are matched by the "identifier" column, not by position.

$WD = "test_scripts/test_data"
$thresholdFile = Join-Path $WD "expect_memory.txt"
. (Join-Path $PSScriptRoot "remeasure.ps1")

$lines  = Get-Content $thresholdFile
$header = $lines[0] -split "`t"
$thrIdx = $header.IndexOf("thr-$Platform")
$fbIdx  = $header.IndexOf($Platform)
if ($thrIdx -ge 0) {
    Write-Host "Using per-platform thresholds: thr-$Platform"
} else {
    Write-Host "No thr-$Platform column; using the shared diff-threshold"
    $thrIdx = $header.IndexOf("diff-threshold")
}
if ($fbIdx -lt 0) {
    Write-Host "WARNING: fallback column '$Platform' not found in $thresholdFile; skipping fallback"
}

$table = @{}
foreach ($line in $lines | Select-Object -Skip 1) {
    $p = $line -split "`t"
    $table[$p[0]] = @{ Threshold = [double]$p[$thrIdx]
                       Fallback  = if ($fbIdx -ge 0) { [double]$p[$fbIdx] } else { $null } }
}

# Memory is column 3 (0-based) of each log: identifier, Command, RealTime, PeakMemory
function Read-Log($path) {
    $h = @{}
    foreach ($line in (Get-Content $path | Select-Object -Skip 1)) {
        $p = $line -split "`t"
        $h[$p[0]] = @{ Command = $p[1]; Value = [double]$p[3] }
    }
    return $h
}
$log2 = Read-Log $IQTree2Log
$log3 = Read-Log $IQTree3Log

$failCount = 0

foreach ($line in (Get-Content $IQTree3Log | Select-Object -Skip 1)) {
    $id = ($line -split "`t")[0]
    if (-not $table.ContainsKey($id)) {
        Write-Host "SKIP $id (no row in $thresholdFile; add one to check it)"
        continue
    }
    $threshold = $table[$id].Threshold
    $reported  = $log3[$id].Value
    $expected  = if ($log2.ContainsKey($id)) { $log2[$id].Value } else { 0 }

    if ($expected -eq 0) {
        if ($null -ne $table[$id].Fallback) {
            $expected = $table[$id].Fallback
            Write-Host "INFO  ${id}: IQ-TREE 2 baseline unavailable, using pre-defined expected value (${expected}MB)"
        } else {
            Write-Host "SKIP $id (IQ-TREE 2 baseline unavailable, no fallback column provided)"
            continue
        }
    }

    $allowed = $expected + $threshold
    $diff    = $reported - $expected

    # Retry once before failing.
    if ($reported -gt $allowed -and $log2.ContainsKey($id)) {
        Write-Host "RETRY $id exceeded (${diff}MB); retrying this command once..."
        $r2 = Measure-Once $log2[$id].Command
        $r3 = Measure-Once $log3[$id].Command
        if ($r2.Ok -and $r3.Ok) {
            $expected = $r2.Mem; $reported = $r3.Mem
            $allowed  = $expected + $threshold
            $diff     = $reported - $expected
            Write-Host "   retry: IQ-TREE2 $($r2.Mem)MB, IQ-TREE3 $($r3.Mem)MB, Diff ${diff}MB"
        } else {
            Write-Host "   retry did not produce a usable measurement; keeping the first result"
        }
    }

    if ($reported -gt $allowed) {
        Write-Host "FAIL $id exceeded the allowed memory usage."
        Write-Host "   Expected: ${expected}MB, Threshold: ${threshold}MB, IQ-TREE3: ${reported}MB, Diff: ${diff}MB"
        $failCount++
    } else {
        Write-Host "PASS $id passed the memory check."
        Write-Host "   Expected: ${expected}MB, Threshold: ${threshold}MB, IQ-TREE3: ${reported}MB, Diff: ${diff}MB"
    }
}

foreach ($id in $table.Keys) {
    if (-not $log3.ContainsKey($id)) {
        Write-Host "WARNING $id : a row exists in $thresholdFile but no command produced it"
    }
}

Write-Host ""
if ($failCount -eq 0) {
    Write-Host "All memory checks passed."
    exit 0
} else {
    Write-Host "$failCount checks failed."
    exit 1
}
