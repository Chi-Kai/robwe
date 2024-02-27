#!/bin/bash
for n in 0.5
do
    beta=$n
    echo $beta
    for i in 100
    do
       save_path="autodl-tmp/Fl-Uchida/00dir0.5-$i-nowater"
       python pflipr-master/FL-Uchida/system/main.py --save_folder_name "$save_path" -datadir "data/cifar10" -lr 0.01 --partition "noniid-labeldir" --dataset "Cifar10" --model "watercnn" --num_classes 10 --beta 0.5 -jr 0.1 -gr 50 -nc $i -wbs 0 -w False 
    done
done