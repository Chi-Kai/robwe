#!/bin/bash
for n in 0.1 0.3 0.5
do
    beta=$n
    echo $beta
    for i in 1
    do
       save_path="./fedipr-exp/vgg/fedipr-cifar100-noniiddir$n-$i"
       python ./robwe/fedipr/main_fedIPR.py  --save_path "$save_path" --lr 0.01 --iid "noniid-labeldir" --dataset "cifar100" --model_name "vgg" --num_classes 100 --epochs 50 --beta $beta --tampered_frac 0 --malicious_frac 0
    done
done

