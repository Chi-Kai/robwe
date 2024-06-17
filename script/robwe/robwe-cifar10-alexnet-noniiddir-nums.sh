#!/bin/bash
for n in 0.5
do
    beta=$n
    echo $beta
    for i in 250
    do
        for j in 2 3 
        do
        save_path="autodl-tmp/halexnetdir-$i-$j"
        python pflipr-master/robwe/robwe.py --save_path "$save_path" --lr 0.01 --partition "noniid-labeldir" --dataset "cifar10" --model "alexnet" --num_classes 10 --beta $beta --epochs 50 --num_users $i --malicious_frac 0 --tampered_frac 0
        done
    done    
done

