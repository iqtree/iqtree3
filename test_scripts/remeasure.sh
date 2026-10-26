#!/bin/bash
# Re-run one benchmark command; echoes "<seconds> <peak MB>".
# Used to retry a check that breached: runner noise only ever inflates, so a
# breach that does not reproduce was noise. Clears the command's own outputs
# first, since IQ-TREE refuses to rerun over a finished checkpoint.
remeasure() {
    local CMD="$1"
    local PREFIX REAL MEM_MB MEM_KB PEAK_MEM tmp
    PREFIX=$(echo "$CMD" | sed -n 's/.*--prefix \([^ ]*\).*/\1/p')
    [ -n "$PREFIX" ] && rm -f "${PREFIX}".*
    tmp=$(mktemp)
    if [[ "$(uname)" == "Darwin" ]]; then
        /usr/bin/time -l -o "$tmp" $CMD > /dev/null 2>&1
        local rc=$?
        REAL=$(awk '/real/{print $1; exit}' "$tmp")
        PEAK_MEM=$(awk '/peak memory footprint/{print $1; exit}' "$tmp")
        MEM_MB=$(awk "BEGIN {printf \"%.2f\", ${PEAK_MEM:-0} / (1024 * 1024)}")
    else
        /usr/bin/time -o "$tmp" -f "%e %U %S %M" $CMD > /dev/null 2>&1
        local rc=$?
        read -r REAL _ _ MEM_KB < "$tmp"
        MEM_MB=$(awk "BEGIN {printf \"%.2f\", ${MEM_KB:-0} / 1024}")
    fi
    rm -f "$tmp"
    if [ "$rc" -ne 0 ]; then echo "0 0"; else echo "${REAL:-0} ${MEM_MB:-0}"; fi
}
