#include <math.h>
#include <stdio.h>

#include "config.h"

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)
#define warp_size 32 //TODO: we need to get device props

/*************************/
/* CURAND INITIALIZATION */
/*************************/
__global__ void initCurand(curandState *state, unsigned long seed, int n_rows){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(x < n_rows) {
        curand_init(seed, x, 0, &state[x]);
    }
}

__global__ void sgd_update(int *indptr, int *indices, const float *data, float *P, float *Q, float *Q_target, 
                           int n_rows, float *user_bias, float *item_bias,
                           float *item_bias_target, curandState *my_curandstate,
                           float global_bias) {

    // One thread per user
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(x < n_rows) {

        // pick a random item y_i
        int low = indptr[x];
        int high = indptr[x+1];
        if(low != high) {
            float myrandf = curand_uniform(&my_curandstate[x]); // random between (0, 1]
            int y_i = (int) ceil(myrandf * (high - low)) - 1 + low;

            // move some reused values to registers
            int y = indices[y_i];
            float ub = user_bias[x];
            float ib = item_bias[y];

            // get the error random item y_i
            float error_y_i = data[y_i] - get_prediction(config::n_factors, &P[x * config::n_factors], &Q[y * config::n_factors], ub, ib, global_bias);

            for(int f = 0; f < config::n_factors; ++f) {
                int p_index = index(x, f, config::n_factors);
                int q_index = index(y, f, config::n_factors);

                // update components
                P[p_index] += config::learning_rate * (error_y_i * Q[q_index] - config::P_reg * P[p_index]);

                // Only update Q if train flag is true
                if(config::is_train) {
                    Q_target[q_index] = Q[q_index] + config::learning_rate * (error_y_i * P[p_index] - config::Q_reg * Q[q_index]);
                }
            }

            // update biases
            user_bias[x] += config::learning_rate * (error_y_i - config::user_bias_reg * ub);

            // Only update item_bias if train flag is true
            if(config::is_train) {
                item_bias_target[y] = ib + config::learning_rate * (error_y_i - config::item_bias_reg * ib);
            }
        }
    }
}