#!/bin/bash
for n in 4
do
    beta=$n
    echo $beta
    for i in 100
    do
       save_path="autodl-tmp/robwe-Uchia/label4-$i-0.001-bn"
       python pflipr-master/robwe-Uchia/robwe.py --save_path "$save_path" --lr 0.001 --partition "noniid-#label${n}" --dataset "cifar10" --model "cnn" --num_classes 10 --beta 0.5 --epochs 50 --num_users $i --embed_dim 100 --malicious_frac 0 --tampered_frac 0
    done
done

