#!/bin/bash
#!/bin/bash
for n in 4
do
    beta=$n
    echo $beta
    for i in 100
    do
       save_path="autodl-tmp/hfl-Ulab4-$i-lr0.001-50"
       python pflipr-master/FL-Uchida/system/fine_attack.py --save_folder_name "$save_path" -datadir "data/cifar10" -lr 0.001 --partition "noniid-#label${n}" --dataset "Cifar10" --model "watercnn"  --layer_type "Conv2d" --num_classes 10 --beta 0.5 -jr 0.1 -gr 50 -nc $i -wbs 50 
    done
done