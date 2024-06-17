#!/bin/bash
for n in 4
do
    beta=$n
    echo $beta
    for i in 100 200 300
    do
        for j in  1
        do
        save_path="autodl-tmp/robwe-replab-bit50-nc$i-$j-50"
        python pflipr-master/robwe/robwe.py --save_path "$save_path" --lr 0.01 --partition "noniid-#label${n}" --dataset "cifar10" --model "cnn" --frac 0.1 --num_classes 10 --epochs 50 --num_users $i --malicious_frac 0 --tampered_frac 0 --embed_dim 0 --rep_bit 50 --use_rep_watermark True --use_watermark False  --detection False
        done
    done    
done