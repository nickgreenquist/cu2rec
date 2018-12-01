#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <algorithm>
#include <vector>
#include <string>

#include "matrix.h"

using namespace cu2rec;

enum ErrorType { MAE, RMSE };

// kernel headers
template <unsigned int block_size>
__global__ void total_loss_kernel(float *in_errors, float *out_errors, int n_errors, ErrorType error_type = RMSE);

// function headers
void calculate_loss_gpu(CudaDenseMatrix* P_d, CudaDenseMatrix* Q_d, int factors, int user_count, int item_count, int num_ratings, CudaCSRMatrix* matrix, float * error_d, float * user_bias, float * item_bias, float global_bias);

float dot_product_sequential(const float *Qi, const float *p, int n);
float calculate_loss_sequential(int factors, int user_count, int item_count, const float * P, const float * Q, const int * indptr, const int * indices, const float * data);
