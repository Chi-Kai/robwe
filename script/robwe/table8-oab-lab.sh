#!/bin/bash
for n in 4
do
    beta=$n
    echo $beta
    for i in 100
    do
        for j in  1
        do
        save_path="autodl-tmp/robwe-table8-oab-nc100-bit50rep50-lab"
        python pflipr-master/robwe/robwe-oab.py --save_path "$save_path" --lr 0.01 --partition "noniid-#label${n}" --dataset "cifar10" --model "cnn" --frac 0.1 --num_classes 10 --epochs 50 --num_users $i --malicious_frac 0 --tampered_frac 0 --embed_dim 50 --rep_bit 50 --use_rep_watermark True --use_watermark True  --detection False
        done
    done    
done