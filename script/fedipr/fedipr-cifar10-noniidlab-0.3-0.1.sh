#!/bin/bash
for n in 4
do
    beta=$n
    echo $beta
    for i in 1
    do
       save_path="autodl-tmp/fedipr/0301lab-$i"
       python pflipr-master/fedipr/main_fedIPR.py  --save_path "$save_path" --lr 0.01 --iid "noniid-#label${n}" --dataset "cifar10" --num_classes 10 --epochs 50 --num_users 100 --num_sign 100  --malicious_frac 0.3 --tampered_frac 0.1
    done
done

