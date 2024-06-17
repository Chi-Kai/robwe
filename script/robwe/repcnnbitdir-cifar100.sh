#!/bin/bash
for n in 0.1 0.3 0.5
do
    beta=$n
    echo $beta
    for i in 0 50 100 150
    do
        for j in  1
        do
        save_path="./robwe-exp/vgg/robwe-repdir$n-bit$i-nc100-$j-50"
        python robwe/robwe/robwe.py --save_path "$save_path" --lr 0.01 --partition "noniid-labeldir" --dataset "cifar100" --model "vgg" --frac 0.1  --beta $n --num_classes 100 --epochs 50 --num_users 100 --malicious_frac 0 --tampered_frac 0 --embed_dim $i --rep_bit 0 
        done
    done    
done