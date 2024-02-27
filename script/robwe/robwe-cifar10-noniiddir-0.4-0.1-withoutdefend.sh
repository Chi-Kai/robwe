#!/bin/bash
for n in 0.5
do
    beta=$n
    echo $beta
    for i in 2 3 
    do
       save_path="autodl-tmp/0401dir0.5-$i"
       python pflipr-master/robwe/robwe.py --save_path "$save_path" --lr 0.01 --partition "noniid-labeldir" --dataset "cifar10" --model "cnn" --num_classes 10 --beta 0.5 --epochs 50 --num_users 100 --malicious_frac 0.4 --tampered_frac 0.1
    done
done

