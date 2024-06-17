for n in 4
do
    beta=$n
    echo $beta
    for i in  200 250 300  
    do
        for j in 1 
        do
        save_path="autodl-tmp/iprcnnlab-bit-$i-$j"
        python pflipr-master/fedipr/fedIPR_bit.py --save_path "$save_path" --lr 0.01 --iid "noniid-#label${n}" --dataset "cifar10" --model_name "cnn" --num_classes 10  --epochs 50 --num_users 100 --num_sign 100 --num_bit $i --malicious_frac 0 --tampered_frac 0
        done
    done    
done