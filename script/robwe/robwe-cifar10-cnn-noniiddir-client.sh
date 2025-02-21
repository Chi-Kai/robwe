#!/bin/bash
for n in 0.5
do
    beta=$n
    echo $beta
    for i in  50  150 200 250
    do
        for j in  2 3 4
        do
        save_path="autodl-tmp/robwe-cnndir-client-b-$i-$j"
        python pflipr-master/robwe/robwe-head.py --save_path "$save_path" --lr 0.01 --partition "noniid-labeldir" --dataset "cifar10" --model "cnn" --num_classes 10 --beta $beta --epochs 50 --num_users $i --malicious_frac 0 --tampered_frac 0 --embed_dim 50  --use_rep_watermark Flase
        done
    done    
done