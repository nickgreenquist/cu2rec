#include <stdexcept>
#include <sstream>
#include <iostream>     // std::cout
#include <math.h>       /* pow */

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "matrix.h"

#define index(i, j, N)  ((i)*(N)) + (j)
#define warp_size 32 //TODO: we need to get device props

using namespace cu2rec;

// PARALLEL
__global__ void loss_kernel(int factors, int user_count, int item_count, const float * P, const float * Q, const int * indptr, 
                            const int * indices, const float * data, float * error, float * user_bias, float * item_bias, float global_bias) {

    
    extern __shared__ float s_memory[];
    float* s_user_bias = (float*)s_memory;

    // TODO: Only load in user_bias values that this block's threads will hit
    // use first warp to load in user_biases
    if(threadIdx.x < warp_size) {
        for(int i = threadIdx.x; i < user_count; i += warp_size) {
            s_user_bias[i] = user_bias[i];
        }
    }
    // sync all threads before accessing any shared memory
    __syncthreads();
    
    // One thread per user
    int u = blockDim.x * blockIdx.x + threadIdx.x;
    if(u < user_count) {
        // get this user's factors into closer memory
        const float * p = &P[u * factors];
        const float ub = s_user_bias[u];

        for (int i = indptr[u]; i < indptr[u + 1]; ++i) {
            int item_id = indices[i];
            error[i] = data[i] - get_prediction(factors, p, &Q[item_id * factors], ub, item_bias[item_id], global_bias);
        }
    }
}

__global__ void total_loss_kernel(float *errors, float *losses, int n_errors, int current_iter, float discount) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i = n_errors / 2; i > 0; i >>= 1) {
        __syncthreads();
        if(x < i) {
            if(i == n_errors / 2) {
                // First iteration
                // Need to square the errors
                errors[x] = pow(errors[x], 2) + pow(errors[x + i], 2);
            } else {
                errors[x] += errors[x + i];
            }
        }
    }
    if(x == 0) {
        // Doing this atomic, in case we want to parallelize this calculation using streams
        atomicAdd(&losses[current_iter], discount * errors[0]);
    }
}

void calculate_loss_gpu(CudaDenseMatrix* P_d, CudaDenseMatrix* Q_d, int factors, int user_count, int item_count, int num_ratings, 
                        CudaCSRMatrix* matrix, float * error_d, float * user_bias,  float * item_bias, float global_bias) {
    int n_threads = 32;
    dim3 dimBlock(n_threads);
    dim3 dimGrid(user_count / n_threads + 1);
    float shared_mem_size = user_count * sizeof(float);
    loss_kernel<<<dimGrid, dimBlock, shared_mem_size>>>(
        factors, user_count, item_count, P_d->data, Q_d->data,
        matrix->indptr, matrix->indices, matrix->data, error_d,
        user_bias, item_bias, global_bias);
    cudaError_t lastError = cudaGetLastError();
    if(cudaSuccess != lastError) {
        printf("ERROR: %s\n", cudaGetErrorName(lastError));
    }
}