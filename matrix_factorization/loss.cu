#include <stdexcept>
#include <sstream>
#include <iostream>     // std::cout
#include <math.h>       /* pow */

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "matrix.h"
#include "util.h"

#define index(i, j, N)  ((i)*(N)) + (j)
#define warp_size 32 //TODO: we need to get device props

using namespace cu2rec;

// PARALLEL
__global__ void loss_kernel(int factors, int user_count, int item_count, const float * P, const float * Q, const int * indptr, 
                            const int * indices, const float * data, float * error, float * user_bias, float * item_bias, float global_bias) {
    extern __shared__ float s_user_bias[];
    
    // One thread per user
    int u = blockDim.x * blockIdx.x + threadIdx.x;
    s_user_bias[threadIdx.x] = user_bias[u];
    if(u < user_count) {
        // get this user's factors into closer memory
        const float * p = &P[u * factors];
        const float ub = s_user_bias[threadIdx.x]; // user_bias[u];

        for (int i = indptr[u]; i < indptr[u + 1]; ++i) {
            int item_id = indices[i];
            error[i] = data[i] - get_prediction(factors, p, &Q[item_id * factors], ub, item_bias[item_id], global_bias);
        }
    }
}

// Inspired by https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// and https://devblogs.nvidia.com/using-shared-memory-cuda-cc/
// Fixes the problems related to data sizes
template <unsigned int block_size>
__global__ void total_loss_kernel(float *in_errors, float *out_errors, int n_errors, ErrorType error_type) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * block_size + tid;
    unsigned int grid_size = block_size * gridDim.x;
    sdata[tid] = 0;
    while (i < n_errors) {
        sdata[tid] += error_type == RMSE ? pow(in_errors[i], 2) : abs(in_errors[i]);
        i += grid_size;
    }
    __syncthreads();
    if (block_size >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (block_size >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (block_size >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    if (tid < block_size / 2) {
        if (block_size >= 64) sdata[tid] += sdata[tid + 32];
        if (block_size >= 32) sdata[tid] += sdata[tid + 16];
        if (block_size >= 16) sdata[tid] += sdata[tid + 8];
        if (block_size >= 8) sdata[tid] += sdata[tid + 4];
        if (block_size >= 4) sdata[tid] += sdata[tid + 2];
        if (block_size >= 2) sdata[tid] += sdata[tid + 1];
    }
    if (tid == 0) out_errors[blockIdx.x] = sdata[0];
}

void calculate_loss_gpu(CudaDenseMatrix* P_d, CudaDenseMatrix* Q_d, int factors, int user_count, int item_count, int num_ratings, 
                        CudaCSRMatrix* matrix, float * error_d, float * user_bias,  float * item_bias, float global_bias) {
    int n_threads = 32;
    dim3 dimBlock(n_threads);
    dim3 dimGrid(user_count / n_threads + 1);
    float shared_mem_size = dimBlock.x * sizeof(float);
    loss_kernel<<<dimGrid, dimBlock, shared_mem_size>>>(
        factors, user_count, item_count, P_d->data, Q_d->data,
        matrix->indptr, matrix->indices, matrix->data, error_d,
        user_bias, item_bias, global_bias);
    CHECK_CUDA(cudaGetLastError());
}