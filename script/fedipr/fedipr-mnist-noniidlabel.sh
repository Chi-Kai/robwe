#!/bin/bash
for n in 2 
do
    beta=$n
    echo $beta
    for i in 10
    do
       save_path=""
       python fedipr/main_fedIPR.py  --save_path "$save_path" --lr 0.01 --iid "noniid-#label${n}" --dataset "mnist" --num_classes 10 --epochs 50
    done
done

