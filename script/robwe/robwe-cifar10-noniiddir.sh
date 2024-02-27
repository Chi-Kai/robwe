#!/bin/bash
for n in 0.5
do
    beta=$n
    echo $beta
    for i in 50
    do
       save_path="autodl-tmp/00dir0.5-$i"
       python pflipr-master/robwe/robwe.py --save_path "$save_path" --lr 0.01 --partition "noniid-labeldir" --dataset "cifar10" --model "cnn" --num_classes 10 --beta 0.5 --epochs 50 --num_users $i --malicious_frac 0 --tampered_frac 0
    done
done

