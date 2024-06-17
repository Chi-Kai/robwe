#!/bin/bash
for n in 0.5
do
    beta=$n
    echo $beta
    for i in 0.3 0.7
    do
        for j in 2 3 4 5 
        do
        save_path="autodl-tmp/bad99dir-$i-$j"
        python pflipr-master/robwe/robwe.py --save_path "$save_path" --lr 0.01 --partition "noniid-labeldir" --dataset "cifar10" --model "cnn" --num_classes 10 --beta 0.5 --epochs 50 --num_users 100 --malicious_frac 0.2 --tampered_frac 0.1 --confidence_level_nor 0.997 --confidence_level_bad $i --bad_nums 7
        done
    done    
done
