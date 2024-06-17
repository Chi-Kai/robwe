#!/bin/bash
for n in 0.5
do
    beta=$n
    echo $beta
    for i in 100
    do
       save_path="autodl-tmp/hrobwe-alexnet-dir0.5prue-$i"
       python pflipr-master/robwe/robwe-normal.py --save_path "$save_path" --lr 0.01 --partition "noniid-labeldir" --dataset "cifar10" --model "alexnet" --num_classes 10 --beta 0.5 --epochs 50 --num_users $i --malicious_frac 0 --tampered_frac 0
    done
done

