#include <stdio.h>

#include "config.h"

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

__global__ void sgd_update(int *indptr, int *indices, float *P, float *Q, float *P_target, float *Q_target, 
                           float *errors, int n_rows, int n_cols, float *user_bias, float *item_bias,
                           float *user_bias_target, float *item_bias_target) {
    // One thread per user
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(x < n_rows) {
        // Loop over all the ratings of the user
        for(int y_i = indptr[x]; y_i < indptr[x + 1]; ++y_i) {
            int y = indices[y_i];
            for(int f = 0; f < config::n_factors; ++f) {
                int p_index = index(x, f, config::n_factors);
                int q_index = index(y, f, config::n_factors);
                // printf("User %d item %d updating P %d (%d, %d) and Q %d (%d, %d)\n", x, y, p_index, x, f, q_index, f, y);

                // Update user and item biases
                float ub_update = config::learning_rate * (errors[y_i] - config::user_bias_reg * user_bias[x]);
                user_bias_target[x] += ub_update;
                float ib_update = config::learning_rate * (errors[y_i] - config::item_bias_reg * item_bias[y]);
                atomicAdd(&item_bias_target[y], ib_update);

                // Update latent factors
                float p_update = config::learning_rate * (errors[y_i] * Q[q_index] - config::P_reg * P[p_index]);
                P_target[p_index] += p_update;
                float q_update = config::learning_rate * (errors[y_i] * P[p_index] - config::Q_reg * Q[q_index]);
                atomicAdd(&Q_target[q_index], q_update);
            }
        }
    }
}