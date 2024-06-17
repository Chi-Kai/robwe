#!/bin/bash
for n in 0.5
do
    beta=$n
    echo $beta
    for i in 4 10
    do
        for j in 1 2 3 4 5 
        do
        save_path="autodl-tmp/num99dir-$i-$j"
        python pflipr-master/robwe/robwe.py --save_path "$save_path" --lr 0.01 --partition "noniid-labeldir" --dataset "cifar10" --model "cnn" --num_classes 10 --beta 0.5 --epochs 50 --num_users 100 --malicious_frac 0.2 --tampered_frac 0.1 --confidence_level_nor 0.997 --confidence_level_bad 0.5 --bad_nums $i
        done
    done    
done