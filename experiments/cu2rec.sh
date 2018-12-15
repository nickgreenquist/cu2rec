#!/usr/bin/env bash

(cd ../matrix_factorization/ && make mf)
mkdir -p results
commit=`git rev-parse --short HEAD`
date=`date '+%Y-%m-%d-%H-%M-%S'`
filename="results/$date-$commit.txt"
datasets=('ml-100k' 'ml-20m')
iterations=(100 500 1000 5000 10000)
factors=(50 300)

for dataset in "${datasets[@]}"; do
    for iteration in "${iterations[@]}"; do
        for factor in "${factors[@]}"; do
            python ../preprocessing/create_config.py exp.cfg -n "$iteration" -f "$factor"
            { time ../matrix_factorization/bin/mf -c exp.cfg "../data/datasets/$dataset/ratings_mapped_train.csv" "../data/datasets/$dataset/ratings_mapped_test.csv" ; } >> $filename 2>&1
            echo "Done with $factor factors with $iteration iterations on $dataset" | tee -a $filename
        done
    done
done
