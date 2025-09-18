param(
    [int]   $ExpectedColumn = 8,                                # 1-based index of the threshold column
    [string]$input_file     = "test_scripts/test_data/expected_runtimes.tsv"
)

$fail_count = 0
$line_num = 0

Get-Content $input_file | ForEach-Object {
    $line = $_.Trim()
    $line_num++

    if ($line_num -eq 1 -or $line -eq "") {
        return  # skip header or empty line
    }

    $columns = $line -split "`t"
    if ($columns.Count -lt 4) {
        Write-Host "Skipping malformed line: $line"
        return
    }

    $iqtree_file = Join-Path "test_scripts/test_data" $columns[0]
    $field_name = $columns[1]
    $expected_value = [double]$columns[$ExpectedColumn]
    $threshold = [double]$columns[2]

    if (-not (Test-Path $iqtree_file)) {
        Write-Host "File not found: ${iqtree_file}"
        return
    }

    $actual_line = Select-String -Path $iqtree_file -Pattern ([regex]::Escape($field_name))
    if (-not $actual_line) {
        Write-Host "Field not found in ${iqtree_file}: ${field_name}"
        return
    }

    $match = [regex]::Match($actual_line.Line, '[-+]?\d+(\.\d+)?([eE][-+]?\d+)?')

    if (-not $match.Success) {
        Write-Host "No numeric value found in line: $($actual_line.Line)"
        return
    }

    $actual_value = [double]$match.Value
    $highest_value = $expected_value + $threshold

    if ($actual_value -le $highest_value) {
        Write-Host "PASS: ${iqtree_file} -- Expected: ${expected_value}, Reported: ${actual_value}, Threshold: ${threshold}"
    } else {
        Write-Host "FAIL: ${iqtree_file} -- Expected: ${expected_value}, Reported: ${actual_value}, Threshold: ${threshold}"
        $fail_count++
    }
}

Write-Host ""
if ($fail_count -eq 0) {
    Write-Host "✅ All runtime checks passed."
} else {
    Write-Host "❌ ${fail_count} checks failed."
    exit 1
}
