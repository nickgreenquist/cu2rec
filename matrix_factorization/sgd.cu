#include <math.h>
#include <stdio.h>

#include "config.h"

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

/** Initializes the random seed for each user.
 */
__global__ void initCurand(curandState *state, unsigned long seed, int n_rows){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(x < n_rows) {
        curand_init(seed, x, 0, &state[x]);
    }
}

/** Kernel for calculating the gradient descent updates. Each thread does one random item per user,
 * and multiple updates to the same item does not stack up, but rather, overwrite each other
 * as mentioned in Hogwild.
 */
__global__ void sgd_update(int *indptr, int *indices, const float *data, float *P, float *Q, float *Q_target, 
                           int n_rows, float *user_bias, float *item_bias,
                           float *item_bias_target, curandState *my_curandstate,
                           float global_bias, int start_user, bool *item_is_updated) {
    // One thread per user
    int x = (blockDim.x * blockIdx.x + threadIdx.x + start_user) % n_rows;
    if(x < n_rows) {

        // pick a random item y_i
        int low = indptr[x];
        int high = indptr[x+1];

        // Only do SGD if the user has at least one item
        if(low != high) {
            float myrandf = curand_uniform(&my_curandstate[x]); // random between (0, 1]
            int y_i = (int) ceil(myrandf * (high - low)) - 1 + low; // random integer between [low, high)

            // move some reused values to registers
            int y = indices[y_i];
            float ub = user_bias[x];
            float ib = item_bias[y];

            // get the error random item y_i
            float error_y_i = data[y_i] - get_prediction(config::n_factors, &P[x * config::n_factors], &Q[y * config::n_factors], ub, ib, global_bias);

            // check if someone already updated this item's feature weights
            bool early_bird = !item_is_updated[y];
            item_is_updated[y] = true;

            // update components
            for(int f = 0; f < config::n_factors; ++f) {
                float P_old = P[index(x, f, config::n_factors)];
                float Q_old = Q[index(y, f, config::n_factors)];

                // update components
                P[index(x, f, config::n_factors)] = P_old + config::learning_rate * (error_y_i * Q_old - config::P_reg * P_old);

                // Only update Q if train flag is true and thread is the early bird
                if(config::is_train && early_bird) {
                    Q_target[index(y, f, config::n_factors)] = Q_old+ config::learning_rate * (error_y_i * P_old - config::Q_reg * Q_old);
                }
            }

            // update user bias
            user_bias[x] += config::learning_rate * (error_y_i - config::user_bias_reg * ub);

            // Only update item_bias if train flag is true and thread is the early bird
            if(config::is_train && early_bird) {
                item_bias_target[y] = ib + config::learning_rate * (error_y_i - config::item_bias_reg * ib);
            }
        }
    }
}