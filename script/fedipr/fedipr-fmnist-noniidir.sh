#!/bin/bash
for n in 0.1 0.3 0.5
do
    beta=$n
    echo $beta
    for i in 10 
    do
       save_path=""
       python fedipr/main_fedIPR.py  --save_path "$save_path" --lr 0.01 --iid "noniid-labeldir" --dataset "fmnist" --num_classes 10 --epochs 50 --beta $beta
    done
done

