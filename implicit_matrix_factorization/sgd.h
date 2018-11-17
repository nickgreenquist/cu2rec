#ifndef CU2REC_SGD
#define CU2REC_SGD

#include "matrix.h"

using namespace cu2rec;

__global__ void sgd_update(int *indptr, int *indices, float *P, float *Q, float *P_target, float *Q_target, int n_factors, float *errors, int n_rows, int n_cols, float learning_rate);

#endif
