#!/bin/bash
# Compare IQ-TREE 3 runtime against the IQ-TREE 2 baseline + threshold.
# When the IQ-TREE 2 baseline is 0 (unsupported command), falls back to the
# pre-defined expected value from expect_runtime.txt if a platform column is given.
# Rows are matched by the "identifier" column, not by position.
#
# Args: $1 = IQ-TREE 2 log file (default: time_log_iqtree2.tsv)
#       $2 = IQ-TREE 3 log file (default: time_log_iqtree3.tsv)
#       $3 = platform name, selects thr-<platform> and the fallback column

iqtree2_log="${1:-time_log_iqtree2.tsv}"
iqtree3_log="${2:-time_log_iqtree3.tsv}"
platform="${3:-}"

WD="test_scripts/test_data"
threshold_file="${WD}/expect_runtime.txt"
# shellcheck source=/dev/null
. "$(dirname "$0")/remeasure.sh"

if [ -n "$platform" ]; then
    if head -1 "$threshold_file" | tr '\t' '\n' | grep -qx "thr-$platform"; then
        echo "Using per-platform thresholds: thr-$platform"
    else
        echo "No thr-$platform column; using the shared diff-threshold"
    fi
    head -1 "$threshold_file" | tr '\t' '\n' | grep -qx "$platform" || \
        echo "WARNING: fallback column '$platform' not found in $threshold_file; skipping fallback"
fi

tmp_join=$(mktemp)

# Join table and both logs on the identifier, in log order.
# Runtime is column 3 of each log (identifier, Command, RealTime, PeakMemory).
awk -F'\t' -v plat="$platform" -v L2="$iqtree2_log" -v L3="$iqtree3_log" '
FILENAME == ARGV[1] {                       # threshold table
    if (FNR == 1) {
        for (i = 1; i <= NF; i++) col[$i] = i
        thr = ("thr-" plat) in col ? col["thr-" plat] : col["diff-threshold"]
        fb  = plat in col ? col[plat] : 0
        next
    }
    threshold[$1] = $thr
    fallback[$1]  = fb ? $fb : ""
    haverow[$1]   = 1
    next
}
FILENAME == L2 { if (FNR > 1) { time2[$1] = $3; cmd2[$1] = $2 } next }
FILENAME == L3 { if (FNR > 1) {
    n++; id[n] = $1; time3[$1] = $3; cmd3[$1] = $2
} next }
END {
    for (i = 1; i <= n; i++) {
        k = id[i]
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", k, haverow[k] ? "Y" : "N",
               threshold[k], fallback[k], time2[k], time3[k], cmd2[k], cmd3[k]
    }
    for (k in haverow) if (!(k in time3)) printf "%s\tORPHAN\t\t\t\t\t\t\n", k
}' "$threshold_file" "$iqtree2_log" "$iqtree3_log" > "$tmp_join"

fail_count=0

while IFS=$'\t' read -r command have threshold fallback iqtree2_val iqtree3_val cmd2 cmd3; do
    if [ "$have" = "ORPHAN" ]; then
        echo "⚠️  $command: a row exists in $threshold_file but no command produced it"
        continue
    fi
    if [ "$have" != "Y" ]; then
        echo "⏭ $command skipped (no row in $threshold_file; add one to check it)"
        continue
    fi

    expected="$iqtree2_val"
    if [ "$(echo "$expected == 0" | bc -l)" = "1" ]; then
        if [ -n "$fallback" ]; then
            expected="$fallback"
            echo "ℹ️  $command: IQ-TREE 2 baseline unavailable, using pre-defined expected value (${expected}s)"
        else
            echo "⏭ $command skipped (IQ-TREE 2 baseline unavailable, no fallback column provided)"
            continue
        fi
    fi

    allowed=$(echo "$expected + $threshold" | bc -l)
    is_exceed=$(echo "$iqtree3_val > $allowed" | bc -l)
    diff=$(echo "$iqtree3_val - $expected" | bc -l)

    # Retry once before failing; costs nothing when everything passes.
    if [ "$is_exceed" = "1" ] && [ -n "$cmd3" ]; then
        echo "↻ $command exceeded (${diff}s); retrying this command once..."
        read -r retry2_time _ <<< "$(remeasure "$cmd2")"
        read -r retry3_time _ <<< "$(remeasure "$cmd3")"
        if [ "$(echo "$retry2_time > 0" | bc -l)" = "1" ] && [ "$(echo "$retry3_time > 0" | bc -l)" = "1" ]; then
            expected="$retry2_time"; iqtree3_val="$retry3_time"
            allowed=$(echo "$expected + $threshold" | bc -l)
            is_exceed=$(echo "$iqtree3_val > $allowed" | bc -l)
            diff=$(echo "$iqtree3_val - $expected" | bc -l)
            echo "   retry: IQ-TREE2 ${retry2_time}s, IQ-TREE3 ${retry3_time}s, Diff ${diff}s"
        else
            echo "   retry did not produce a usable measurement; keeping the first result"
        fi
    fi

    if [ "$is_exceed" = "1" ]; then
        echo "❌ $command exceeded the allowed runtime usage."
        echo "   Expected: ${expected}s, Threshold: ${threshold}s, IQ-TREE3: ${iqtree3_val}s, Diff: ${diff}s"
        ((fail_count++))
    else
        echo "✅ $command passed the runtime check."
        echo "   Expected: ${expected}s, Threshold: ${threshold}s, IQ-TREE3: ${iqtree3_val}s, Diff: ${diff}s"
    fi
done < "$tmp_join"

rm -f "$tmp_join"

if [ "$fail_count" -eq 0 ]; then
    echo "✅ All runtime checks passed."
    exit 0
else
    echo "❌ $fail_count checks failed."
    exit 1
fi
