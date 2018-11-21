# cu2rec: CUDA Meets Recommender Systems

## Data
1. Download movielens data and save in `data` folder
2. Link: https://grouplens.org/datasets/movielens/

## Compiling Code
1. SSH into Prince or Cuda using NYU credentials
2. `srun -t5:00:00 --mem=30000 --gres=gpu:1 --pty /bin/bash`
3. `module load cuda/9.2.88`
4. `make`

## Running Tests
1. `cd tests`
2. `make`
3. If you want to run all tests, `make run_all`
4. Otherwise, `bin/test_{}`