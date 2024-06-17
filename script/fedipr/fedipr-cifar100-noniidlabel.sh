#!/bin/bash
for n in 2 4 6
do
    beta=$n
    echo $beta
    for i in 10
    do
       save_path="./fedipr-exp/fedipr-cifar10-noniidlabel$n"
       python ./robwe/fedipr/main_fedIPR.py  --save_path "$save_path" --lr 0.01 --iid "noniid-#label${n}" --dataset "cifar100" --num_classes 100 --epochs 50 --tampered_frac 0 --malicious_frac 0
    done
done

