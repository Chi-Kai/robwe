#!/bin/bash
#!/bin/bash
for n in 4
do
    beta=$n
    echo $beta
    for i in 150
    do
       save_path="autodl-tmp/Udir0.5-alex-lr0.01-nc-$i-bits100-test50"
       python pflipr-master/FL-Uchida/system/main.py --save_folder_name "$save_path" -datadir "data/cifar10" -lr 0.001 --partition "noniid-labeldir" --dataset "Cifar10" --model "alexnet" --layer_type "fc" --num_classes 10 --beta 0.5 -jr 0.1 -gr 50 -nc $i -wbs 100
    done
done