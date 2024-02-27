#!/bin/bash
for n in 2
do
    beta=$n
    echo $beta
    for i in 10 
    do
       save_path=""
       python robwe/robwe.py --save_path "$save_path" --lr 0.001 --partition "noniid-#label${n}" --dataset "mnist" --num_classes 10 --epochs 50
    done
done

