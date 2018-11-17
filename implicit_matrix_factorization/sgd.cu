#include <cuda.h>

#include "matrix.h"

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

__global__ void sgd_update(CudaCSRMatrix *mat, float *P, float *Q, float *P_target, float *Q_target, int n_factors, float *errors, int n_rows, int n_cols, float learning_rate) {
    // One thread per user
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n_rows) {
        // Loop over all the ratings of the user
        for(int y_i = mat->indptr[x]; y_i < mat->indptr[x + 1]; ++y_i) {
            int y = mat->indices[y_i];
            for(int f = 0; f < n_factors; ++f) {
                float p_update = learning_rate * errors[y_i] * Q[index(f, y, n_cols)];
                P_target[index(f, y, n_cols)]] += p_update
                float q_update = learning_rate * errors[y_i] * P[index(x, f, n_factors)];
                Q_target[index(x, f, n_factors)] += q_update;
            }
        }
    }
}