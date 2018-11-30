#ifndef CU2REC_TRAINING
#define CU2REC_TRAINING

#include <cuda.h>

#include "config.h"
#include "matrix.h"

using namespace cu2rec;

void train(CudaCSRMatrix* train_matrix, CudaCSRMatrix* test_matrix, config::Config* cfg, float **P_ptr, float **Q_ptr, float *Q, float **losses_ptr,
           float **user_bias_ptr, float **item_bias_ptr, float *item_bias, float global_bias);
void train(CudaCSRMatrix* train_matrix, CudaCSRMatrix* test_matrix, config::Config* cfg, float **P_ptr, float **Q_ptr, float **losses_ptr,
           float **user_bias_ptr, float **item_bias_ptr, float global_bias);

#endif
