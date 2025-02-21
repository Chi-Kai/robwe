#!/bin/bash
for n in 0.5
do
    beta=$n
    echo $beta
    for i in 1 2 3 4 5 6 7 8 9 10
    do
       save_path="autodl-tmp/h0403dir0.5-$i"
       python pflipr-master/robwe/robwe.py --save_path "$save_path" --lr 0.01 --partition "noniid-labeldir" --dataset "cifar10" --model "cnn" --num_classes 10 --beta 0.5 --epochs 50 --num_users 100 --malicious_frac 0.4 --tampered_frac 0.3
    done
done

