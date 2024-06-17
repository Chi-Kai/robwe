#!/bin/bash
for n in 4
do
    beta=$n
    echo $beta
    for i in  200 250 300
    do
        for j in  1
        do
        save_path="autodl-tmp/robwe-cnnlab-bit-b-$i-$j"
        python pflipr-master/robwe/robwe-head.py --save_path "$save_path" --lr 0.01 --partition "noniid-#label${n}" --dataset "cifar10" --model "alexnet" --num_classes 10 --epochs 50 --num_users 100 --malicious_frac 0 --tampered_frac 0 --embed_dim $i --use_rep_watermark Flase
        done
    done    
done