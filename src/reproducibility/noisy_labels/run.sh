#!/bin/bash

DATA_CONFIGS=(
    clean
#    aggre
#    worse
#    random1
#    random2
#    random3
)
METHOD_CONFIGS=(
#    CAL
#    CE 
#    co_teaching 
#    co_teaching_plus 
#    cores2 
#    divide_mix 
#    ELR 
#    ELR_plus
#    GCE
    jocor
#    PES
#    PES_semi
#    SOP
#    SOP_plus
#    volminnet
)
SEEDS=(
    1337
#    42
#    1
#    0
#    10086
)

for DATA_CONFIG in ${DATA_CONFIGS[@]}
do
    for METHOD_CONFIG in ${METHOD_CONFIGS[@]}
    do
        for SEED in ${SEEDS[@]}
        do
            python main.py $METHOD_CONFIG $DATA_CONFIG --seed $SEED
        done
    done
done