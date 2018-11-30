#ifndef CU2REC_SGD
#define CU2REC_SGD

#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#include <cuda.h>

using namespace cu2rec;

__global__ void initCurand(curandState *state, unsigned long seed, int n_rows);
__global__ void sgd_update(int *indptr, int *indices, float *P, float *Q, float *P_target, float *Q_target, 
                           float *errors, int n_rows, int n_cols, float *user_bias, float *item_bias,
                           float *user_bias_target, float *item_bias_target, curandState *my_curandstate);

#endif
