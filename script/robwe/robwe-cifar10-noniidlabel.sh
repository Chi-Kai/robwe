#!/bin/bash
for n in 4
do
    beta=$n
    echo $beta
    for i in 100
    do
       save_path="autodl-tmp/hrobwelab-alexnet-prue-$i"
       python pflipr-master/robwe/robwe-normal.py --save_path "$save_path" --lr 0.01 --partition "noniid-#label${n}" --dataset "cifar10" --model "alexnet" --num_classes 10 --epochs 50 --num_users $i --malicious_frac 0 --tampered_frac 0
    done
done

