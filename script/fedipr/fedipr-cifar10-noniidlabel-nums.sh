#!/bin/bash
for n in 4
do
    beta=$n
    echo $beta
    for i in 50 150 250
    do
       save_path="autodl-tmp/fedipr/00label0.5-$i-2"
       python pflipr-master/fedipr/main_fedIPR.py  --save_path "$save_path" --lr 0.01 --iid "noniid-#label${n}" --dataset "cifar10" --num_classes 10 --epochs 50  --malicious_frac 0 --tampered_frac 0 --num_users $i  --num_sign $i 
    done
done
