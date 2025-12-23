#!/bin/bash
#$ -M ctaguchi@nd.edu
#$ -m abe
#$ -pe smp 8
#$ -N eval-mms-mix-adapt-max3
#$ -l gpu_card=1
#$ -q gpu

export HF_HOME="/afs/crc.nd.edu/group/nlp/07/ctaguchi/.cache/"
LANGUAGES=(
    aln
    bew
    bxk
    cgg
    el-CY
    hch
    kcn
    koo
    led
    lke
    lth
    meh
    mmc
    pne
    ruc
    rwm
    sco
    tob
    top
    ttj
    ukv
)

# debug
START=0
LEN=1

for LANG in "${LANGUAGES[@]:START:LEN}"; do
    uv run python main_experiment.py \
        --language $LANG \
        --model max3 \
        --use_local_model \
        --beam_width 50
    mail -s "${LANG} done." ctaguchi@nd.edu < stats/stats_${LANG}.json
done
    
