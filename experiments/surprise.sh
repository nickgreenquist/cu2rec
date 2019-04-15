#!/usr/bin/env bash

mkdir -p results
commit=`git rev-parse --short HEAD`
date=`date '+%Y-%m-%d-%H-%M-%S'`
filename="results/$date-$commit.txt"
datasets=('goodreads-2018')
iterations=(100 500 1000 5000 10000)
factors=(50 300)

for dataset in "${datasets[@]}"; do
    for iteration in "${iterations[@]}"; do
        for factor in "${factors[@]}"; do
            { time python run_surprise.py "../data/datasets/$dataset/ratings.csv" -n "$iteration" -f "$factor" ; } >> $filename 2>&1
            echo "Done with $factor factors with $iteration iterations on $dataset" | tee -a $filename
        done
    done
done
