#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <algorithm>
#include <vector>
#include <string>

#include "matrix.h"

using namespace cu2rec;

// kernel headers
__global__ void total_loss_kernel(float *errors, float *losses, int n_errors, int current_iter);

// function headers
void calculate_loss_gpu(CudaDenseMatrix* P_d, CudaDenseMatrix* Q_d, int factors, int user_count, int item_count, int num_ratings, CudaCSRMatrix* matrix, float * error_d, float * user_bias, float * item_bias);

float dot_product_sequential(const float *Qi, const float *p, int n);
float calculate_loss_sequential(int factors, int user_count, int item_count, const float * P, const float * Q, const int * indptr, const int * indices, const float * data);
