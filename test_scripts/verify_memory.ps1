# Compare IQ-TREE 3 memory usage against IQ-TREE 2 baseline + threshold.
# When IQ-TREE 2 baseline is 0 (unsupported command), falls back to the
# pre-defined expected value from expected_memory.tsv if a platform column is given.
#
# Args: $IQTree2Log    = IQ-TREE 2 log file
#       $IQTree3Log    = IQ-TREE 3 log file
#       $FallbackColumn = platform column name in expected_memory.tsv for fallback (optional)
param (
    [string] $IQTree2Log     = "time_log_iqtree2.tsv",
    [string] $IQTree3Log     = "time_log_iqtree3.tsv",
    [string] $FallbackColumn = ""
)

$WD = "test_scripts/test_data"
$thresholdFile = Join-Path $WD "expected_memory.tsv"

# Read thresholds (command + diff-threshold only)
$thresholdLines = Get-Content $thresholdFile | Select-Object -Skip 1
$thresholds = foreach ($line in $thresholdLines) {
    $parts = $line -split "`t"
    [PSCustomObject]@{ Command = $parts[0]; Threshold = [double]$parts[1] }
}

# Resolve fallback column index
$fallbackValues = @()
if ($FallbackColumn -ne "") {
    $header = (Get-Content $thresholdFile -TotalCount 1) -split "`t"
    $colIdx = $header.IndexOf($FallbackColumn)
    if ($colIdx -lt 0) {
        Write-Host "WARNING: fallback column '$FallbackColumn' not found in $thresholdFile; skipping fallback"
        $FallbackColumn = ""
    } else {
        $fallbackValues = foreach ($line in $thresholdLines) { [double]($line -split "`t")[$colIdx] }
    }
}

# Read memory (column index 2 = PeakMemory)
$iqtree2Lines = Get-Content $IQTree2Log | Select-Object -Skip 1
$iqtree2Mem = foreach ($line in $iqtree2Lines) { [double]($line -split "`t")[2] }

$iqtree3Lines = Get-Content $IQTree3Log | Select-Object -Skip 1
$iqtree3Mem = foreach ($line in $iqtree3Lines) { [double]($line -split "`t")[2] }

$failCount = 0

for ($i = 0; $i -lt $thresholds.Count; $i++) {
    $command   = $thresholds[$i].Command
    $threshold = $thresholds[$i].Threshold
    $expected  = $iqtree2Mem[$i]
    $reported  = $iqtree3Mem[$i]

    if ($expected -eq 0) {
        if ($FallbackColumn -ne "" -and $fallbackValues.Count -gt $i) {
            $expected = $fallbackValues[$i]
            Write-Host "ℹ️  ${command}: IQ-TREE 2 baseline unavailable, using pre-defined expected value (${expected}MB)"
        } else {
            Write-Host "⏭ $command skipped (IQ-TREE 2 baseline unavailable, no fallback column provided)"
            continue
        }
    }

    $allowed = $expected + $threshold
    $diff    = $reported - $expected

    if ($reported -gt $allowed) {
        Write-Host "❌ $command exceeded the allowed memory usage."
        Write-Host "   Expected: ${expected}MB, Threshold: ${threshold}MB, IQ-TREE3: ${reported}MB, Diff: ${diff}MB"
        $failCount++
    } else {
        Write-Host "✅ $command passed the memory check."
        Write-Host "   Expected: ${expected}MB, Threshold: ${threshold}MB, IQ-TREE3: ${reported}MB, Diff: ${diff}MB"
    }
}

Write-Host ""

if ($failCount -eq 0) {
    Write-Host "✅ All memory checks passed."
    exit 0
} else {
    Write-Host "❌ $failCount checks failed."
    exit 1
}
