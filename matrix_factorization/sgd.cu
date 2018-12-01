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

__global__ void sgd_update(int *indptr, int *indices, const float *data,float *P, float *Q, float *Q_target, 
                           float *errors, int n_rows, int n_cols, float *user_bias, float *item_bias,
                           float *item_bias_target, curandState *my_curandstate,
                            float global_bias) {

    extern __shared__ float s_memory[];
    float* s_user_bias = (float*)s_memory;

    // TODO: Only load in user_bias values that this block's threads will hit
    // use first warp to load in user_biases
    if(threadIdx.x < warp_size) {
        for(int i = threadIdx.x; i < n_rows; i += warp_size) {
            s_user_bias[i] = user_bias[i];
        }
    }
    // sync all threads before accessing any shared memory
    __syncthreads();

    // One thread per user
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(x < n_rows) {

        // pick a random item y_i
        int low = indptr[x];
        int high = indptr[x+1];
        float myrandf = curand_uniform(&my_curandstate[x]); // random between (0, 1]
        int y_i = (int) ceil(myrandf * (high - low)) - 1 + low;

        // get the error random item y_i
        int item_id = indices[y_i];
        float error_y_i = data[y_i] - get_prediction(config::n_factors, &P[x * config::n_factors], &Q[item_id * config::n_factors], s_user_bias[x], item_bias[item_id], global_bias);

        int y = indices[y_i];
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
        user_bias[x] += config::learning_rate * (error_y_i - config::user_bias_reg * user_bias[x]);

        // Only update item_bias if train flag is true
        if(config::is_train) {
            item_bias_target[y] = item_bias[y] + config::learning_rate * (error_y_i - config::item_bias_reg * item_bias[y]);
        }
    }
}