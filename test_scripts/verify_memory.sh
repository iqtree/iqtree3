#!/bin/bash
# expect the first argument to be the column name : contain what platform(OS) is running the script
expected_column="$1"

WD="test_scripts/test_data"
input_file="${WD}/expected_memory.tsv"
reported_file="time_log.tsv"
selected_columns_file="${WD}/selected_columns.tsv"

# Get the column index (1-based) of the expected column name
col_index=$(head -1 "$input_file" | tr '\t' '\n' | awk -v col="$expected_column" '{if ($0 == col) print NR}')

# Check if column was found
if [[ -z "$col_index" ]]; then
  echo "Column '$expected_column' not found in $input_file"
  exit 1
fi

cut -f1,2,"$col_index" "$input_file" > "${selected_columns_file}"
final_file="${WD}/combined_with_reported.tsv"

# assuming the reported file and expected file have the same order of commands
temp_reported_column=$(mktemp)
cut -f3 "$reported_file" > "$temp_reported_column"
paste "$selected_columns_file" "$temp_reported_column" > "$final_file"
rm -f "$temp_reported_column"


fail_count=0

# Skip header
while IFS=$'\t' read -r command threshold expected reported; do
    allowed=$(echo "$expected + $threshold" | bc -l)
    is_exceed=$(echo "$reported > $allowed" | bc -l)

    if [ "$is_exceed" = "1" ]; then
        diff=$(echo "$reported - $expected" | bc -l)
        echo "❌ $command exceeded the allowed memory usage."
        echo "Expected: $expected MB, Threshold: $threshold MB, Reported: $reported MB, Difference: $diff MB"
        ((fail_count++))
    else
        echo "✅ $command passed the memory check."
        diff=$(echo "$reported - $expected" | bc -l)
        echo "Expected: $expected MB, Threshold: $threshold MB, Reported: $reported MB, Difference: $diff MB"
    fi
done < <(tail -n +2 "$final_file")

if [ "$fail_count" -eq 0 ]; then
    echo "✅ All memory checks passed."
else
    echo "❌ $fail_count checks failed."
    exit 1
fi
