# cu2rec: CUDA Meets Recommender Systems

## Helpful Commands for Compile CUDA Code:
1. SSH into Prince
2. `srun -t5:00:00 --mem=30000 --gres=gpu:1 --pty /bin/bash`
3. `module load cuda/9.2.88`
4. `nvcc -o matrix -arch=sm_52 matrix.cu`