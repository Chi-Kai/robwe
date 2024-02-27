#!/bin/bash
for n in 0.5
do
    beta=$n
    echo $beta
    for i in 1 2 3 
    do
       save_path="autodl-tmp/fedipr/00dir0.5-$i"
       python pflipr-master/fedipr/main_fedIPR.py  --save_path "$save_path" --lr 0.01 --iid "noniid-labeldir" --dataset "cifar10" --num_classes 10 --epochs 50 --beta $beta --malicious_frac 0 --tampered_frac 0
    done
done

