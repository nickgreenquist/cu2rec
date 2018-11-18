#ifndef CU2REC_TRAINING
#define CU2REC_TRAINING

#include <cuda.h>
#include "matrix.h"

using namespace cu2rec;

void train(CudaCSRMatrix* matrix, int n_iterations, int n_factors, float learning_rate, int seed,
           float **P_ptr, float **Q_ptr, float **losses_ptr, float **user_bias_ptr, float **item_bias_ptr);

#endif
