#!/bin/bash
for n in 4
do
    beta=$n
    echo $beta
    for i in  100
    do    
        save_path="autodl-tmp/hrobwelab0.5prue-$i"
        python pflipr-master/robwe/pruning_attack.py --save_path "$save_path" --lr 0.01  --partition "noniid-#label${n}" --dataset "cifar10" --model "cnn" --num_classes 10 --beta $beta --epochs 50 --num_users 100 --malicious_frac 0 --tampered_frac 0 --embed_dim 100  
    done    
done

