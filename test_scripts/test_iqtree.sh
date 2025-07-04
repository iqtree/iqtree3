#!/bin/bash

SEED=73073

LOGFILE="time_log.tsv"
WD=test_scripts/test_data
BUILD_DIR="build"

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
        # macOS
        /usr/bin/time -l "$@" 2> tmp_time.txt
        REAL=$(grep "real" tmp_time.txt | awk '{print $1}')
        USER=$(grep "user" tmp_time.txt | awk '{print $1}')
        SYS=$(grep "sys" tmp_time.txt | awk '{print $1}')
        PEAK_MEM=$(grep "peak memory footprint" tmp_time.txt | awk '{print $1}')
        MEM_MB=$(awk "BEGIN {printf \"%.2f\", $PEAK_MEM / (1024 * 1024)}")
    else
        # Linux
        /usr/bin/time -f "%e %U %S %M" "$@" 2> tmp_time.txt
        read REAL USER SYS MEM_KB < tmp_time.txt
        MEM_MB=$(awk "BEGIN {printf \"%.2f\", $MEM_KB / 1024}")
    fi

    # Append to log
    echo -e "$CMD\t$REAL\t$MEM_MB" >> "$LOGFILE"

    rm -f tmp_time.txt
}

run_timed ${BUILD_DIR}/iqtree3 -s ${WD}/turtle.fa -B 1000 -T 1 -seed $SEED

run_timed ${BUILD_DIR}/iqtree3 -s ${WD}/turtle.fa -p ${WD}/turtle.nex -B 1000 -T 1 -seed $SEED

run_timed ${BUILD_DIR}/iqtree3 -s ${WD}/turtle.fa -p ${WD}/turtle.nex -B 1000 -T 1 -m MFP+MERGE -rcluster 10 --prefix ${WD}/turtle.merge -seed $SEED

cat ${WD}/turtle.fa.treefile ${WD}/turtle.nex.treefile > ${WD}/turtle.trees
run_timed ${BUILD_DIR}/iqtree3 -s ${WD}/turtle.fa -p ${WD}/turtle.merge.best_scheme.nex -z ${WD}/turtle.trees -zb 10000 -au -n 0 --prefix ${WD}/turtle.test -seed $SEED -T 1

run_timed ${BUILD_DIR}/iqtree3 -s ${WD}/turtle.fa -m GTR+F+I+R3+T -te ${WD}/turtle.trees -T 1 --prefix ${WD}/turtle.mix -seed $SEED

run_timed ${BUILD_DIR}/iqtree3 -s ${WD}/turtle.fa -p ${WD}/turtle.nex.best_scheme.nex -z ${WD}/turtle.trees -n 0 -wpl --prefix ${WD}/turtle.wpl -seed $SEED -T 1

run_timed ${BUILD_DIR}/iqtree3 -s ${WD}/turtle.fa -S ${WD}/turtle.nex --prefix ${WD}/turtle.loci -T 1 -seed $SEED

run_timed ${BUILD_DIR}/iqtree3 -t ${WD}/turtle.nex.treefile --gcf ${WD}/turtle.loci.treefile -s ${WD}/turtle.fa --scf 100 -seed $SEED -T 1

run_timed ${BUILD_DIR}/iqtree3 -t ${WD}/turtle.fa.treefile --gcf ${WD}/turtle.loci.treefile -s ${WD}/turtle.fa --scf 100 -seed $SEED -T 1

# link-exchange-rates model

run_timed ${BUILD_DIR}/iqtree3 -s ${WD}/turtle.fa -m "MIX{GTR+FO,GTR+FO}" --link-exchange-rates --prefix ${WD}/turtle.mix.link -seed $SEED -T 1

run_timed ${BUILD_DIR}/iqtree3 -s ${WD}/turtle.fa -m "MIX{GTR{1,1,1,1,1,1}+FO,GTR{1,1,1,1,1,1}+FO}" --link-exchange-rates --prefix ${WD}/turtle.mix.jc.link -seed $SEED -T 1

run_timed ${BUILD_DIR}/iqtree3 -s ${WD}/turtle.fa -p ${WD}/turtle.nex -g ${WD}/turtle.constr.tree --prefix ${WD}/turtle.nex.constr -T 1 -seed $SEED

run_timed ${BUILD_DIR}/iqtree3 -s ${WD}/turtle.fa -p ${WD}/turtle.nex -g ${WD}/turtle.constr.tree2 -B 1000 -alrt 1000 --prefix ${WD}/turtle.nex.constr2 -T 1 -seed $SEED

run_timed ${BUILD_DIR}/iqtree3 -s ${WD}/turtle.fa -m "MIX+MF" --prefix ${WD}/turtle.mixfinder -T 1 -seed $SEED


## amino acid test cases
# the data set is a subset of the turtle data set
echo "Running amino acid test cases..."
AA_FASTA=${WD}/turtle_aa.fasta
AA_NEX=${WD}/turtle_aa.nex
AA_prefix=${WD}/turtle_aa

run_timed ${BUILD_DIR}/iqtree3 -s $AA_FASTA -B 1000 -T 1 -seed $SEED

run_timed ${BUILD_DIR}/iqtree3 -s $AA_FASTA -p $AA_NEX -B 1000 -T 1 -seed $SEED

run_timed ${BUILD_DIR}/iqtree3 -s $AA_FASTA -p $AA_NEX -B 1000 -T 1 -m MFP+MERGE -rcluster 10 --prefix ${WD}/turtle_aa.merge -seed $SEED

cat $AA_FASTA.treefile $AA_NEX.treefile > ${WD}/turtle_aa.trees
run_timed ${BUILD_DIR}/iqtree3 -s $AA_FASTA -p ${WD}/turtle_aa.merge.best_scheme.nex -z ${WD}/turtle_aa.trees -zb 10000 -au -n 0 --prefix ${WD}/turtle_aa.test -seed $SEED -T 1

run_timed ${BUILD_DIR}/iqtree3 -s $AA_FASTA -p $AA_NEX.best_scheme.nex -z ${WD}/turtle_aa.trees -n 0 -wpl --prefix ${WD}/turtle_aa.wpl -seed $SEED -T 1

run_timed ${BUILD_DIR}/iqtree3 -s $AA_FASTA -S $AA_NEX --prefix ${WD}/turtle_aa.loci -T 1 -seed $SEED

run_timed ${BUILD_DIR}/iqtree3 -t $AA_NEX.treefile --gcf ${WD}/turtle_aa.loci.treefile -s $AA_FASTA --scf 100 -seed $SEED -T 1

run_timed ${BUILD_DIR}/iqtree3 -t $AA_FASTA.treefile --gcf ${WD}/turtle_aa.loci.treefile -s $AA_FASTA --scf 100 -seed $SEED -T 1

run_timed ${BUILD_DIR}/iqtree3 -s $AA_FASTA -m "MIX{LG+F,WAG+F}" --prefix ${WD}/turtle_aa.mix -seed $SEED -T 1

run_timed ${BUILD_DIR}/iqtree3 -s $AA_FASTA -p $AA_NEX -g ${WD}/turtle.constr.tree --prefix $AA_NEX.constr -T 1 -seed $SEED

run_timed ${BUILD_DIR}/iqtree3 -s $AA_FASTA -p $AA_NEX -g ${WD}/turtle.constr.tree2 -B 1000 -alrt 1000 --prefix $AA_NEX.constr2 -T 1 -seed $SEED

