# cu2rec: CUDA Meets Recommender Systems

## Data
1. Download movielens data and save in `data` folder
2. Link: https://grouplens.org/datasets/movielens/

## Compiling Code
1. SSH into Prince or Cuda using NYU credentials
2. `srun -t5:00:00 --mem=30000 --gres=gpu:1 --pty /bin/bash`
3. `module load cuda/9.2.88`
4. `make`

## Training
1. `make mf`
2. `bin/mf -c <config_file> <ratings_file>`

## Getting recommendations for a user
1. Make sure you get the user data into the same ratings format as MovieLens.
2. `make predict`
3. `bin/predict -c <config_file> -i <trained_item_bias_file> -g <trained_global_bias_file> -q <trained_Q_file> <ratings_file>`

## Running Tests
1. `cd tests`
2. `make`
3. If you want to run all tests, `make run_all`
4. Otherwise, `bin/test_{}`
