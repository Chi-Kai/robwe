#!/bin/bash
for n in 0.5
do
    beta=$n
    echo $beta
    for i in 1 2 3 
    do
       save_path="autodl-tmp/fedipr/0301dir0.5-right-$i"
       python pflipr-master/fedipr/main_fedIPR.py  --save_path "$save_path" --lr 0.01 --iid "noniid-labeldir" --dataset "cifar10" --num_classes 10 --epochs 50 --num_users 100 --num_sign 100 --beta $beta --malicious_frac 0.3 --tampered_frac 0.1
    done
done

