#!/bin/bash
for n in 0.1 0.3 0.5
do
    beta=$n
    echo $beta
    for i in 10 
    do
       save_path=""
       python robwe/robwe.py --save_path "$save_path" --lr 0.01 --partition "noniid-labeldir" --dataset "fmnist" --num_classes 10 --beta $beta --epochs 50 --num_users 100
    done
done

