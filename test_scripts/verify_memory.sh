#!/bin/bash
# Compare IQ-TREE 3 memory usage against IQ-TREE 2 baseline + threshold.
# When IQ-TREE 2 baseline is 0 (unsupported command), falls back to the
# pre-defined expected value from expected_memory.tsv if a platform column is given.
#
# Args: $1 = IQ-TREE 2 log file (default: time_log_iqtree2.tsv)
#       $2 = IQ-TREE 3 log file (default: time_log_iqtree3.tsv)
#       $3 = platform column name in expected_memory.tsv for fallback (optional)

iqtree2_log="${1:-time_log_iqtree2.tsv}"
iqtree3_log="${2:-time_log_iqtree3.tsv}"
fallback_column="${3:-}"

WD="test_scripts/test_data"
threshold_file="${WD}/expected_memory.tsv"

tmp_thresholds=$(mktemp)
tmp_fallback=$(mktemp)
tmp_iqtree2=$(mktemp)
tmp_iqtree3=$(mktemp)

# Command and diff-threshold columns (columns 1 and 2)
tail -n +2 "$threshold_file" | cut -f1,2 > "$tmp_thresholds"

# Pre-defined fallback expected values for the given platform column
if [ -n "$fallback_column" ]; then
    col_index=$(head -1 "$threshold_file" | tr '\t' '\n' | awk -v col="$fallback_column" '{if ($0 == col) print NR}')
    if [ -z "$col_index" ]; then
        echo "WARNING: fallback column '$fallback_column' not found in $threshold_file; skipping fallback"
        fallback_column=""
    else
        tail -n +2 "$threshold_file" | cut -f"$col_index" > "$tmp_fallback"
    fi
fi

# Memory is column 3 of each log
tail -n +2 "$iqtree2_log" | cut -f3 > "$tmp_iqtree2"
tail -n +2 "$iqtree3_log" | cut -f3 > "$tmp_iqtree3"

fail_count=0
row=0

while IFS=$'\t' read -r command threshold iqtree2_val iqtree3_val; do
    ((row++))
    expected="$iqtree2_val"

    # Fall back to pre-defined value when IQ-TREE 2 baseline is unavailable
    if [ "$(echo "$expected == 0" | bc -l)" = "1" ]; then
        if [ -n "$fallback_column" ]; then
            expected=$(sed -n "${row}p" "$tmp_fallback")
            echo "ℹ️  $command: IQ-TREE 2 baseline unavailable, using pre-defined expected value (${expected}MB)"
        else
            echo "⏭ $command skipped (IQ-TREE 2 baseline unavailable, no fallback column provided)"
            continue
        fi
    fi

    allowed=$(echo "$expected + $threshold" | bc -l)
    is_exceed=$(echo "$iqtree3_val > $allowed" | bc -l)
    diff=$(echo "$iqtree3_val - $expected" | bc -l)

    if [ "$is_exceed" = "1" ]; then
        echo "❌ $command exceeded the allowed memory usage."
        echo "   Expected: ${expected}MB, Threshold: ${threshold}MB, IQ-TREE3: ${iqtree3_val}MB, Diff: ${diff}MB"
        ((fail_count++))
    else
        echo "✅ $command passed the memory check."
        echo "   Expected: ${expected}MB, Threshold: ${threshold}MB, IQ-TREE3: ${iqtree3_val}MB, Diff: ${diff}MB"
    fi
done < <(paste "$tmp_thresholds" "$tmp_iqtree2" "$tmp_iqtree3")

rm -f "$tmp_thresholds" "$tmp_fallback" "$tmp_iqtree2" "$tmp_iqtree3"

if [ "$fail_count" -eq 0 ]; then
    echo "✅ All memory checks passed."
else
    echo "❌ $fail_count checks failed."
    exit 1
fi
