#!/bin/bash

SEED=73073

IQTREE_BIN="${1:-build/iqtree3}"
LOGFILE="${2:-time_log.tsv}"
WD=test_scripts/test_data
# All output files go to OUT_DIR; input data always comes from WD.
# Pass a different OUT_DIR for the IQ-TREE 2 baseline run to avoid
# overwriting IQ-TREE 3 output files.
OUT_DIR="${3:-${WD}}"

mkdir -p "${OUT_DIR}"

# Initialize TSV file with header
echo -e "Command\tRealTime(s)\tPeakMemory(MB)" > "$LOGFILE"

run_timed() {
    local CMD="$*"
    echo -e "\n================ RUNNING ================="
    echo "$CMD"
    echo "=========================================="

    local OS=$(uname)
    local REAL USER SYS PEAK_MEM MEM_MB

    if [[ "$OS" == "Darwin" ]]; then
        /usr/bin/time -l -o tmp_time.txt "$@"
        local exit_code=$?
        REAL=$(grep "real" tmp_time.txt | awk '{print $1}')
        USER=$(grep "user" tmp_time.txt | awk '{print $1}')
        SYS=$(grep "sys" tmp_time.txt | awk '{print $1}')
        PEAK_MEM=$(grep "peak memory footprint" tmp_time.txt | awk '{print $1}')
        MEM_MB=$(awk "BEGIN {printf \"%.2f\", $PEAK_MEM / (1024 * 1024)}")
    else
        /usr/bin/time -o tmp_time.txt -f "%e %U %S %M" "$@"
        local exit_code=$?
        read REAL USER SYS MEM_KB < tmp_time.txt
        MEM_MB=$(awk "BEGIN {printf \"%.2f\", $MEM_KB / 1024}")
    fi

    if [ "$exit_code" -ne 0 ]; then
        echo "WARNING: command exited with code $exit_code, logging 0 for time and memory"
        REAL=0
        MEM_MB=0
    fi

    # Append to log
    echo -e "$CMD\t$REAL\t$MEM_MB" >> "$LOGFILE"

    rm -f tmp_time.txt
}

run_timed ${IQTREE_BIN} -s ${WD}/turtle.fa -B 1000 -T 1 -seed $SEED \
    --prefix ${OUT_DIR}/turtle.fa

run_timed ${IQTREE_BIN} -s ${WD}/turtle.fa -p ${WD}/turtle.nex -B 1000 -T 1 -seed $SEED \
    --prefix ${OUT_DIR}/turtle.nex

run_timed ${IQTREE_BIN} -s ${WD}/turtle.fa -p ${WD}/turtle.nex -B 1000 -T 1 -m MFP+MERGE -rcluster 10 --prefix ${OUT_DIR}/turtle.merge -seed $SEED

cat ${OUT_DIR}/turtle.fa.treefile ${OUT_DIR}/turtle.nex.treefile > ${OUT_DIR}/turtle.trees
run_timed ${IQTREE_BIN} -s ${WD}/turtle.fa -p ${OUT_DIR}/turtle.merge.best_scheme.nex -z ${OUT_DIR}/turtle.trees -zb 10000 -au -n 0 --prefix ${OUT_DIR}/turtle.test -seed $SEED -T 1

run_timed ${IQTREE_BIN} -s ${WD}/turtle.fa -m GTR+F+I+R3+T -te ${OUT_DIR}/turtle.trees -T 1 --prefix ${OUT_DIR}/turtle.mix -seed $SEED

run_timed ${IQTREE_BIN} -s ${WD}/turtle.fa -p ${OUT_DIR}/turtle.nex.best_scheme.nex -z ${OUT_DIR}/turtle.trees -n 0 -wpl --prefix ${OUT_DIR}/turtle.wpl -seed $SEED -T 1

run_timed ${IQTREE_BIN} -s ${WD}/turtle.fa -S ${WD}/turtle.nex --prefix ${OUT_DIR}/turtle.loci -T 1 -seed $SEED

run_timed ${IQTREE_BIN} -t ${OUT_DIR}/turtle.nex.treefile --gcf ${OUT_DIR}/turtle.loci.treefile -s ${WD}/turtle.fa --scf 100 -seed $SEED -T 1 \
    --prefix ${OUT_DIR}/turtle.nex.cf

run_timed ${IQTREE_BIN} -t ${OUT_DIR}/turtle.fa.treefile --gcf ${OUT_DIR}/turtle.loci.treefile -s ${WD}/turtle.fa --scf 100 -seed $SEED -T 1 \
    --prefix ${OUT_DIR}/turtle.fa.cf

# link-exchange-rates model

run_timed ${IQTREE_BIN} -s ${WD}/turtle.fa -m "MIX{GTR+FO,GTR+FO}" --link-exchange-rates --prefix ${OUT_DIR}/turtle.mix.link -seed $SEED -T 1

run_timed ${IQTREE_BIN} -s ${WD}/turtle.fa -m "MIX{GTR{1,1,1,1,1,1}+FO,GTR{1,1,1,1,1,1}+FO}" --link-exchange-rates --prefix ${OUT_DIR}/turtle.mix.jc.link -seed $SEED -T 1

run_timed ${IQTREE_BIN} -s ${WD}/turtle.fa -p ${WD}/turtle.nex -g ${WD}/turtle.constr.tree --prefix ${OUT_DIR}/turtle.nex.constr -T 1 -seed $SEED

run_timed ${IQTREE_BIN} -s ${WD}/turtle.fa -p ${WD}/turtle.nex -g ${WD}/turtle.constr.tree2 -B 1000 -alrt 1000 --prefix ${OUT_DIR}/turtle.nex.constr2 -T 1 -seed $SEED

run_timed ${IQTREE_BIN} -s ${WD}/turtle.fa -m "MIX+MF" --prefix ${OUT_DIR}/turtle.mixfinder -T 1 -seed $SEED


## amino acid test cases
echo "Running amino acid test cases..."
AA_FASTA=${WD}/turtle_aa.fasta
AA_NEX=${WD}/turtle_aa.nex

run_timed ${IQTREE_BIN} -s $AA_FASTA -B 1000 -T 1 -seed $SEED \
    --prefix ${OUT_DIR}/turtle_aa.fasta

run_timed ${IQTREE_BIN} -s $AA_FASTA -p $AA_NEX -B 1000 -T 1 -seed $SEED \
    --prefix ${OUT_DIR}/turtle_aa.nex

run_timed ${IQTREE_BIN} -s $AA_FASTA -p $AA_NEX -B 1000 -T 1 -m MFP+MERGE -rcluster 10 --prefix ${OUT_DIR}/turtle_aa.merge -seed $SEED

cat ${OUT_DIR}/turtle_aa.fasta.treefile ${OUT_DIR}/turtle_aa.nex.treefile > ${OUT_DIR}/turtle_aa.trees
run_timed ${IQTREE_BIN} -s $AA_FASTA -p ${OUT_DIR}/turtle_aa.merge.best_scheme.nex -z ${OUT_DIR}/turtle_aa.trees -zb 10000 -au -n 0 --prefix ${OUT_DIR}/turtle_aa.test -seed $SEED -T 1

run_timed ${IQTREE_BIN} -s $AA_FASTA -p ${OUT_DIR}/turtle_aa.nex.best_scheme.nex -z ${OUT_DIR}/turtle_aa.trees -n 0 -wpl --prefix ${OUT_DIR}/turtle_aa.wpl -seed $SEED -T 1

run_timed ${IQTREE_BIN} -s $AA_FASTA -S $AA_NEX --prefix ${OUT_DIR}/turtle_aa.loci -T 1 -seed $SEED

run_timed ${IQTREE_BIN} -t ${OUT_DIR}/turtle_aa.nex.treefile --gcf ${OUT_DIR}/turtle_aa.loci.treefile -s $AA_FASTA --scf 100 -seed $SEED -T 1 \
    --prefix ${OUT_DIR}/turtle_aa.nex.cf

run_timed ${IQTREE_BIN} -t ${OUT_DIR}/turtle_aa.fasta.treefile --gcf ${OUT_DIR}/turtle_aa.loci.treefile -s $AA_FASTA --scf 100 -seed $SEED -T 1 \
    --prefix ${OUT_DIR}/turtle_aa.fasta.cf

run_timed ${IQTREE_BIN} -s $AA_FASTA -m "MIX{LG+F,WAG+F}" --prefix ${OUT_DIR}/turtle_aa.mix -seed $SEED -T 1

run_timed ${IQTREE_BIN} -s $AA_FASTA -p $AA_NEX -g ${WD}/turtle.constr.tree --prefix ${OUT_DIR}/turtle_aa.nex.constr -T 1 -seed $SEED

run_timed ${IQTREE_BIN} -s $AA_FASTA -p $AA_NEX -g ${WD}/turtle.constr.tree2 -B 1000 -alrt 1000 --prefix ${OUT_DIR}/turtle_aa.nex.constr2 -T 1 -seed $SEED
