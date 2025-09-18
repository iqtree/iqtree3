#!/bin/bash

input_file="test_scripts/test_data/expect_ans.txt"
fail_count=0
line_num=0

while IFS=$'\t' read -r iqtree_file field_name expected_value threshold || [ -n "$iqtree_file" ]; do
    ((line_num++))

    # Skip header (first line)
    if [ "$line_num" -eq 1 ]; then
        continue
    fi

    iqtree_file="test_scripts/test_data/${iqtree_file}"

    if [ ! -f "$iqtree_file" ]; then
        echo "File not found: $iqtree_file"
        continue
    fi

    # Look for the line containing the field name
    report_line=$(grep -F "$field_name" "$iqtree_file")
    if [ -z "$report_line" ]; then
        echo "Field not found in $iqtree_file: $field_name"
        continue
    fi

    # Extract the first numeric value from the matched line
    report_value=$(echo "$report_line" | grep -Eo '[-]?[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?' | head -n1)
    if [ -z "$report_value" ]; then
        echo "No numeric value found in line: $report_line"
        continue
    fi

    # Compute absolute difference
    diff=$(echo "$report_value - $expected_value" | bc -l)
    abs_diff=$(echo "$diff" | awk '{print ($1 < 0) ? -$1 : $1}')
    result=$(echo "$abs_diff <= $threshold" | bc -l)

    if [ "$result" -eq 1 ]; then
        echo "PASS: $iqtree_file -- Expected: ${expected_value}, Reported: ${report_value}, Abs-diff: ${abs_diff}, Threshold: $threshold"
    else
        echo "FAIL: $iqtree_file ($field_name)"
        echo "  Expected: ${expected_value}, Reported: ${report_value}, Abs-diff: ${abs_diff}, Threshold: $threshold"
        ((fail_count++))
    fi
done < "$input_file"

echo
if [ "$fail_count" -eq 0 ]; then
    echo "✅ All checks passed."
else
    echo "❌ $fail_count checks failed."
    exit 1
fi
