# for n in 4
# do
#     beta=$n
#     echo $beta
#     for i in  150 250
#     do
#         for j in 3
#         do
#         save_path="autodl-tmp/ipr—alexnet—lab-$i-$j"
#         python pflipr-master/fedipr/fedIPR.py --save_path "$save_path" --lr 0.01 --iid "noniid-#label${n}" --dataset "cifar10" --model_name "alexnet" --num_classes 10 --beta $beta --epochs 50 --num_users $i  --num_sign $i --malicious_frac 0 --tampered_frac 0
#         done
#     done    
# done
for n in 4
do
    beta=$n
    echo $beta
    for i in  150 250
    do
        for j in 5
        do
        save_path="autodl-tmp/iprcnnlab-$i-$j"
        python pflipr-master/fedipr/main_fedIPR.py --save_path "$save_path" --lr 0.01 --iid "noniid-#label${n}" --dataset "cifar10" --model_name "cnn" --num_classes 10 --beta $beta --epochs 50 --num_users $i  --num_sign $i --malicious_frac 0 --tampered_frac 0
        done
    done    
done