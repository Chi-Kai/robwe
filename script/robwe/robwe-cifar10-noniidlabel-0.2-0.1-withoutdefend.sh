#!/bin/bash
for n in 4
do
    beta=$n
    echo $beta
    for i in 1 2 3
    do
       save_path="autodl-tmp/0201label4-$i"
       python pflipr-master/robwe/robwe.py --save_path "$save_path" --lr 0.01 --partition "noniid-#label${n}" --dataset "cifar10" --model "cnn" --num_classes 10  --epochs 50 --num_users 100 --malicious_frac 0.2 --tampered_frac 0.1
    done
done
