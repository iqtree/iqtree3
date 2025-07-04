param (
    [int]   $ExpectedColumn = 8
)

$WD = "test_scripts/test_data"
$expectedFile = Join-Path $WD "expected_memory.tsv"
$reportedFile = "time_log.tsv"
$selectedColumnsFile = Join-Path $WD "selected_columns.tsv"
$reportedColumnFile = "$env:TEMP\reported_column.tsv"
$finalFile = Join-Path $WD "combined_with_reported.tsv"

# Read expected: Command, Threshold, ExpectedColumn
$expectedLines = Get-Content $expectedFile | Select-Object -Skip 1
$selectedColumns = foreach ($line in $expectedLines) {
    $parts = $line -split "`t"
    "$($parts[0])`t$($parts[1])`t$($parts[$ExpectedColumn - 1])"
}
$selectedColumns | Set-Content $selectedColumnsFile

# Read reported column from time_log.tsv (column 3 = RealTime, 4 = Memory)
$reportedLines = Get-Content $reportedFile | Select-Object -Skip 1
$reportedColumn = foreach ($line in $reportedLines) {
    $parts = $line -split "`t"
    $parts[2]  # column 3 = memory (0-based index)
}
$reportedColumn | Set-Content $reportedColumnFile

# Combine both files
$combinedLines = @()
for ($i = 0; $i -lt $selectedColumns.Count; $i++) {
    $combinedLines += "$($selectedColumns[$i])`t$($reportedColumn[$i])"
}
$combinedLines | Set-Content $finalFile

# Now compare
Write-Host "`nCommand`tExpected`tThreshold`tActual`tExceededBy"
$failCount = 0
$finalLines = Get-Content $finalFile

foreach ($line in $finalLines) {
    $parts = $line -split "`t"
    $command = $parts[0]
    $threshold = [double]$parts[1]
    $expected = [double]$parts[2]
    $reported = [double]$parts[3]

    $allowed = $expected + $threshold
    $exceededBy = $reported - $expected

    if ($reported -gt $allowed) {
        Write-Host "❌ $command exceeded the allowed usage."
        Write-Host "Expected: $expected MB, Threshold: $threshold MB, Reported: $reported MB"
        $failCount++
    } else {
        Write-Host "✅ $command passed the check."
        Write-Host "Expected: $expected MB, Threshold: $threshold MB, Reported: $reported MB"
    }
}

Write-Host ""

if ($failCount -eq 0) {
    Write-Host "✅ All runtime/memory checks passed."
    exit 0
} else {
    Write-Host "❌ $failCount checks failed."
    exit 1
}
