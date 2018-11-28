#include <stdio.h>

#include "config.h"

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

/*************************/
/* CURAND INITIALIZATION */
/*************************/
__global__ void initCurand(curandState *state, unsigned long seed){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(seed, x, 0, &state[x]);
}

__global__ void sgd_update(int *indptr, int *indices, float *P, float *Q, float *P_target, float *Q_target, 
                           float *errors, int n_rows, int n_cols, float *user_bias, float *item_bias,
                           float *user_bias_target, float *item_bias_target, float global_bias, float * data, curandState *my_curandstate) {
    // One thread per user
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(x < n_rows) {

        // pick a random y_i
        // int low = indptr[x];
        // int high = indptr[x+1];
        // float myrandf = curand_uniform(&my_curandstate[x]);
        // myrandf *= (high - low + 0.999999);
        // myrandf += low;
        // int y_i = (int)truncf(myrandf);
        // printf("y_i: %d\n", y_i);

        // Loop over all the ratings of the user
        for(int y_i = indptr[x]; y_i < indptr[x + 1]; ++y_i) {
            int y = indices[y_i];
            for(int f = 0; f < config::n_factors; ++f) {
                int p_index = index(x, f, config::n_factors);
                int q_index = index(y, f, config::n_factors);

                // Update P
                P_target[p_index] += config::learning_rate * (errors[y_i] * Q[q_index] - config::P_reg * P[p_index]);

                // Only update Q if train flag is true
                if(config::is_train) {
                    // float temp = Q[q_index];
                    // Q_target[q_index] = temp + config::learning_rate * (errors[y_i] * P[p_index] - config::Q_reg * Q[q_index]);
                    atomicAdd(&Q_target[q_index], config::learning_rate * (errors[y_i] * P[p_index] - config::Q_reg * Q[q_index]));
                }
            }

            // update biases
            user_bias_target[x] += config::learning_rate * (errors[y_i] - config::user_bias_reg * user_bias[x]);
            if(config::is_train) {
                // float temp = item_bias[y];
                // item_bias_target[y] = temp + config::learning_rate * (errors[y_i] - config::item_bias_reg * item_bias[y]);
                atomicAdd(&item_bias_target[y], config::learning_rate * (errors[y_i] - config::item_bias_reg * item_bias[y]));
            }
        }
    }
}