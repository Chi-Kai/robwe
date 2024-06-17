#!/bin/bash
for n in 4
do
    beta=$n
    echo $beta
    for i in 100 200 300
    do
        for j in  1
        do
        save_path="autodl-tmp/rep/robwe-alex-repdir-bit$i-nc100-$j-50"
        python pflipr-master/robwe/robwe.py --save_path "$save_path" --lr 0.01 --partition "noniid-labeldir" --dataset "cifar10" --model "alexnet" --frac 0.1  --beta 0.5 --num_classes 10 --epochs 50 --num_users 100 --malicious_frac 0 --tampered_frac 0 --embed_dim 0 --rep_bit $i --use_rep_watermark True --use_watermark False  --detection False
        done
    done    
done