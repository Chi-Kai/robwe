#!/bin/bash
for n in 4
do
    beta=$n
    echo $beta
    for i in 0.997 0.955 0.683
    do
        for j in 2 3 4 5
        do
        save_path="autodl-tmp/nor99lab-$i-$j"
        python pflipr-master/robwe/robwe.py --save_path "$save_path" --lr 0.01 --partition "noniid-#label${n}" --dataset "cifar10" --model "cnn" --num_classes 10  --epochs 50 --num_users 100 --malicious_frac 0.2 --tampered_frac 0.1 --confidence_level_nor $i --confidence_level_bad 0.5 --bad_nums 7
        done
    done    
done

