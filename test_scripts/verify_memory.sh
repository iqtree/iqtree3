#!/bin/bash
expected_column="$1"

WD="test_scripts/test_data"
input_file="${WD}/expected_memory.tsv"
reported_file="time_log.tsv"
selected_columns_file="${WD}/selected_columns.tsv"
cut -f1,2,"$expected_column" "$input_file" > "${selected_columns_file}"
final_file="${WD}/combined_with_reported.tsv"

# assuming the reported file and expected file have the same order of commands
cut -f3 "$reported_file" > /tmp/reported_column.tsv
paste "$selected_columns_file" /tmp/reported_column.tsv > "$final_file"


fail_count=0

echo -e "Command\tExpected\tThreshold\tActual\tExceededBy"

# Skip header
while IFS=$'\t' read -r command threshold expected reported; do
    allowed=$(echo "$expected + $threshold" | bc -l)
    is_exceed=$(echo "$reported > $allowed" | bc -l)

    if [ "$is_exceed" = "1" ]; then
        diff=$(echo "$reported - $expected" | bc -l)
        echo "❌ $command exceeded the allowed memory usage."
        echo "Expected: $expected MB, Threshold: $threshold MB, Reported: $reported MB"
        ((fail_count++))
    else
        echo "✅ $command passed the memory check."
        diff=$(echo "$reported - $expected" | bc -l)
        echo "Expected: $expected MB, Threshold: $threshold MB, Reported: $reported MB"
    fi
done < <(tail -n +2 "$final_file")

if [ "$fail_count" -eq 0 ]; then
    echo "✅ All runtime checks passed."
else
    echo "❌ $fail_count checks failed."
    exit 1
fi
