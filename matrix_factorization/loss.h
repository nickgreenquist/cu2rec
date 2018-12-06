#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <algorithm>
#include <vector>
#include <string>
#include <tuple>

#include "matrix.h"

using namespace cu2rec;

enum ErrorType { MAE, RMSE };

// kernel headers
template <unsigned int block_size>
__global__ void total_loss_kernel(float *in_errors, float *out_errors, int n_errors, ErrorType error_type = RMSE);

// function headers
std::tuple<float, float> get_error_metrics_cpu(float *errors, float *errors_device, int n_errors);
std::tuple<float, float> get_error_metrics_gpu(float *in_errors, float *out_errors, float *out_errors_host, int n_errors, int grid_size, int block_size);
void calculate_loss_gpu(CudaDenseMatrix* P_d, CudaDenseMatrix* Q_d, config::Config* cfg, int user_count, int item_count, int num_ratings, CudaCSRMatrix* matrix, float * error_d, float * user_bias, float * item_bias, float global_bias);
