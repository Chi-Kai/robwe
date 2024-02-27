#!/bin/bash
for n in 4
do
    beta=$n
    echo $beta
    for i in 50 150 250
    do
       save_path="autodl-tmp/00label4-$i-2"
       python pflipr-master/robwe/robwe.py --save_path "$save_path" --lr 0.01 --partition "noniid-#label${n}" --dataset "cifar10" --model "cnn" --num_classes 10 --epochs 50 --num_users $i --malicious_frac 0 --tampered_frac 0
    done
done

