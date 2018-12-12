#ifndef CU2REC_SGD
#define CU2REC_SGD

#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#include <cuda.h>

using namespace cu2rec;

__global__ void initCurand(curandState *state, unsigned long seed, int n_rows);
__global__ void sgd_update(int *indptr, int *indices, const float *data, float *P, float *Q, float *Q_target, 
                           int n_rows, float *user_bias, float *item_bias,
                           float *item_bias_target, curandState *my_curandstate,
                           float global_bias, int start_user);

#endif