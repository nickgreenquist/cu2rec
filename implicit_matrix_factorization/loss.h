#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <algorithm>
#include <vector>
#include <string>

#include "matrix.h"

//function headers
float calculate_loss_gpu(int factors, int user_count, int item_count, const float * P, const float * Q, cu2rec::CudaCSRMatrix* matrix);

float dot_product_sequential(const float *Qi, const float *p, int n);
float calculate_loss_sequential(int factors, int user_count, int item_count, const float * P, const float * Q, const int * indptr, const int * indices, const float * data);