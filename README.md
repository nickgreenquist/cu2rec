# cu2rec: CUDA Meets Recommender Systems

## Compiling Code
1. SSH into Prince or Cuda using NYU credentials
2. `srun -t5:00:00 --mem=30000 --gres=gpu:1 --pty /bin/bash`
3. `module load cuda/9.2.88`
4. `make`

## Running Tests
1. `make` will create test executables
2. run with `./test_{}`