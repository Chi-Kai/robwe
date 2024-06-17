for n in 0.5
do
    beta=$n
    echo $beta
    for i in  200 250 300
    do
        for j in  1
        do
        save_path="autodl-tmp/iprcnndir-bit-$i-$j"
        python pflipr-master/fedipr/fedIPR_bit.py --save_path "$save_path" --lr 0.01 --iid "noniid-labeldir" --dataset "cifar10" --model_name "cnn" --num_classes 10 --beta $beta --epochs 50 --num_bit $i --num_users 100  --num_sign 100 --malicious_frac 0 --tampered_frac 0
        done
    done    
done