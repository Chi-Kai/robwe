for n in 0.5
do
    beta=$n
    echo $beta
    for i in 50 100 150 200 250
    do
        for j in  1 2 
        do
        save_path="autodl-tmp/ipralexnetdirclient-$i-$j"
        python pflipr-master/fedipr/fedIPR.py --save_path "$save_path" --lr 0.01 --iid "noniid-labeldir" --dataset "cifar10" --model_name "alexnet" --num_classes 10 --beta $beta --epochs 50 --num_users $i  --num_sign $i --malicious_frac 0 --tampered_frac 0
        done
    done    
done