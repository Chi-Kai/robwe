#!/bin/bash
for n in 0.5
do
    beta=$n
    echo $beta
    for i in 100
    do

       #save_path="autodl-tmp/robwe-Uchia/hlabel4-$i-0.001"
       save_path="autodl-tmp/robwe-Uchia/hdir0.5-$i-100bit-0.001"
       #save_path="/root/autodl-tmp/robwe-Uchia/dir0.5-100-100bit-0.001"
       python pflipr-master/robwe-Uchia/pruning_attack.py --save_path "$save_path" --lr 0.001 --partition "noniid-labeldir" --dataset "cifar10" --model "cnn" --num_classes 10 --beta 0.5 --epochs 50 --num_users $i --malicious_frac 0 --tampered_frac 0 --embed_dim 100
    done
done

