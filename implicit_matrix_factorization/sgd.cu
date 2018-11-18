#include <cuda.h>
#include <stdio.h>

#include "matrix.h"

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

__global__ void sgd_update(int *indptr, int *indices, float *P, float *Q, float *P_target, float *Q_target, int n_factors, float *errors, int n_rows, int n_cols, float learning_rate) {
    // One thread per user
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(x < n_rows) {
        // Loop over all the ratings of the user
        for(int y_i = indptr[x]; y_i < indptr[x + 1]; ++y_i) {
            int y = indices[y_i];
            for(int f = 0; f < n_factors; ++f) {
                int p_index = index(x, f, n_factors);
                int q_index = index(f, y, n_cols);
                // printf("User %d item %d updating P %d (%d, %d) and Q %d (%d, %d)\n", x, y, p_index, x, f, q_index, f, y);
                float p_update = learning_rate * errors[y_i] * Q[q_index];
                P_target[p_index] += p_update;
                float q_update = learning_rate * errors[y_i] * P[p_index];
                atomicAdd(&Q_target[q_index], q_update);
            }
        }
    }
}