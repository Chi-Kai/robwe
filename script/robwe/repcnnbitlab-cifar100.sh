#!/bin/bash
for n in 2 4 6
do
    beta=$n
    echo $beta
    for i in 0 50 100 150
    do
        for j in  1
        do
        save_path="./robwe-exp/robwe-replab$n-bit$i-nc100-$j-50"
        python robwe/robwe/robwe.py --save_path "$save_path" --lr 0.01 --partition "noniid-#label${n}" --dataset "cifar100" --model "cnn" --frac 0.1 --num_classes 100 --epochs 50 --num_users 100 --malicious_frac 0 --tampered_frac 0 --embed_dim $i --rep_bit 0 --use_rep_watermark False --use_watermark True  --detection False
        done
    done    
done