# #!/bin/bash
# for n in 4
# do
#     beta=$n
#     echo $beta  
#     for i in 100
#     do
#        save_path="autodl-tmp/hfl-Ulab4-$i-lr0.001"
#        python pflipr-master/FL-Uchida/system/prune_model.py --save_folder_name "$save_path" -data "Cifar10"  --model "watercnn" --partition "noniid-#label${n}"  -lr 0.001 --num_classes 10  -nc $i -wbs 100 
       
#     done
# done
#!/bin/bash
for n in 0.5
do
    beta=$n
    echo $beta  
    for i in 100
    do
       save_path="autodl-tmp/hfl-Udir0.5-$i-lr0.001"
       python pflipr-master/FL-Uchida/system/prune_model.py --save_folder_name "$save_path" -data "Cifar10"  --model "watercnn" --partition "noniid-labeldir" --beta 0.5 -lr 0.001 --num_classes 10  -nc $i -wbs 100 
       
    done
done


