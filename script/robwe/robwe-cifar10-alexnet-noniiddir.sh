#!/bin/bash
for n in 250
do
    for i in 2 3 4 5
    do
       save_path="autodl-tmp/halexnet00dir0.5nc50-$i"
       python pflipr-master/robwe/robwe.py --save_path "$save_path" --lr 0.01 --partition "noniid-labeldir" --dataset "cifar10" --model "alexnet" --num_classes 10 --beta 0.5 --epochs 50 --num_users 50 --malicious_frac 0 --tampered_frac 0
    done
done

