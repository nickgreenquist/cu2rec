#include <stdexcept>
#include <sstream>
#include <iostream>     // std::cout
#include <math.h>       /* pow */
#include <tuple>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "matrix.h"
#include "util.h"

using namespace cu2rec;

/** Calculates the losses per item.
 * Each thread does all the items for a single user.
 * Puts the losses in the error array.
 */
__global__ void loss_kernel(int factors, int user_count, int item_count, const float * P, const float * Q, const int * indptr, 
                            const int * indices, const float * data, float * error, float * user_bias, float * item_bias, float global_bias) {
    
    // One thread per user
    int u = blockDim.x * blockIdx.x + threadIdx.x;
    if(u < user_count) {
        // Get this user's factors and bias
        const float * p = &P[u * factors];
        const float ub = user_bias[u];

        // Loop over all items of user
        for (int i = indptr[u]; i < indptr[u + 1]; ++i) {
            int item_id = indices[i];
            error[i] = data[i] - get_prediction(factors, p, &Q[item_id * factors], ub, item_bias[item_id], global_bias);
        }
    }
}

/** Driver function for calculating the losses per item.
 * Puts the losses in the error_d array.
 */
void calculate_loss_gpu(CudaDenseMatrix* P_d, CudaDenseMatrix* Q_d, config::Config* cfg, int user_count, int item_count, int num_ratings, 
                        CudaCSRMatrix* matrix, float * error_d, float * user_bias,  float * item_bias, float global_bias) {
    dim3 dimBlock(cfg->n_threads);
    dim3 dimGrid(user_count / cfg->n_threads + 1);
    loss_kernel<<<dimGrid, dimBlock>>>(
        cfg->n_factors, user_count, item_count, P_d->data, Q_d->data,
        matrix->indptr, matrix->indices, matrix->data, error_d,
        user_bias, item_bias, global_bias);
    CHECK_CUDA(cudaGetLastError());
}

/** Function template for adding up the total loss.
 * Uses reduction to collect all losses into out_errors, which is small
 * (number of blocks) and can be transferred to the host memory and summed in CPU.
 * Inspired by https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 * and https://devblogs.nvidia.com/using-shared-memory-cuda-cc/
 * Fixes the problems related to data sizes, so it can be used with any length data.
 */
template <unsigned int block_size>
__global__ void total_loss_kernel(float *in_errors, double *out_errors, int n_errors, ErrorType error_type) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * block_size + tid;
    unsigned int grid_size = block_size * gridDim.x;
    sdata[tid] = 0;
    // Each thread does n_errors / grid_size work to start, before reduction
    while (i < n_errors) {
        // Each error is actual_rating - predicted_rating.
        // We want its square for RMSE and its absolute value for MAE
        sdata[tid] += error_type == RMSE ? pow(in_errors[i], 2) : abs(in_errors[i]);
        i += grid_size;
    }
    __syncthreads();

    // Start reduction. This is an unrolled loop from 512 to 1.
    // Note that any if statements related to the block size are
    // completely skipped because of templating
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
    if (block_size >= 64) {
        if (tid < 32) {
            sdata[tid] += sdata[tid + 32];
        }
        __syncthreads();
    }
    // if block_size is 1, compiler will complain about unneeded unsigned
    // int comparison (tid) to a value of 0
    if (!(block_size == 1) && tid < block_size / 2) {
        if (block_size >= 32) {
            sdata[tid] += sdata[tid + 16];
            __syncthreads();
        }
        if (block_size >= 16) {
            sdata[tid] += sdata[tid + 8];
            __syncthreads();
        }
        if (block_size >= 8) {
            sdata[tid] += sdata[tid + 4];
            __syncthreads();
        }
        if (block_size >= 4) {
            sdata[tid] += sdata[tid + 2];
            __syncthreads();
        }
        if (block_size >= 2) {
            sdata[tid] += sdata[tid + 1];
            __syncthreads();
        }
    }
    // All the errors in the block are now reduced to the first
    // index of shared data
    if (tid == 0) out_errors[blockIdx.x] = sdata[0];
}

/** Calculates the total Mean Absolute Error and Root Mean Squared Error on CPU.
 */
std::tuple<float, float> get_error_metrics_cpu(float *errors, float *errors_device, int n_errors) {
    cudaMemcpy(errors, errors_device, n_errors * sizeof(float), cudaMemcpyDeviceToHost);
    double mae = 0.0;
    double rmse = 0.0;
    for(int k = 0; k <  n_errors; k++) {
        mae += abs(errors[k]);
        rmse += errors[k] * errors[k];
    }
    mae /= n_errors;
    rmse = sqrt(rmse / n_errors);
    return std::make_tuple((float)mae, (float)rmse);
}

/** Convenience function for adding up the total loss.
 * The block size must be a power of 2, up to 512.
 * Since the block_size in template needs to be known at compile time,
 * this is a workaround for making it variable.
 */
float calculate_error_metric_gpu(float *in_errors, double *out_errors, double *out_errors_host, int n_errors, int grid_size, int block_size, ErrorType error_type) {
    switch(block_size) {
        case 512:
            total_loss_kernel<512><<<grid_size, block_size, 512 * sizeof(double)>>>(in_errors, out_errors, n_errors, error_type);
            break;
        case 256:
            total_loss_kernel<256><<<grid_size, block_size, 256 * sizeof(double)>>>(in_errors, out_errors, n_errors, error_type);
            break;
        case 128:
            total_loss_kernel<128><<<grid_size, block_size, 128 * sizeof(double)>>>(in_errors, out_errors, n_errors, error_type);
            break;
        case 64:
            total_loss_kernel< 64><<<grid_size, block_size,  64 * sizeof(double)>>>(in_errors, out_errors, n_errors, error_type);
            break;
        case 32:
            total_loss_kernel< 32><<<grid_size, block_size,  32 * sizeof(double)>>>(in_errors, out_errors, n_errors, error_type);
            break;
        case 16:
            total_loss_kernel< 16><<<grid_size, block_size,  16 * sizeof(double)>>>(in_errors, out_errors, n_errors, error_type);
            break;
        case 8:
            total_loss_kernel<  8><<<grid_size, block_size,   8 * sizeof(double)>>>(in_errors, out_errors, n_errors, error_type);
            break;
        case 4:
            total_loss_kernel<  4><<<grid_size, block_size,   4 * sizeof(double)>>>(in_errors, out_errors, n_errors, error_type);
            break;
        case 2:
            total_loss_kernel<  2><<<grid_size, block_size,   2 * sizeof(double)>>>(in_errors, out_errors, n_errors, error_type);
            break;
        case 1:
            total_loss_kernel<  1><<<grid_size, block_size,   1 * sizeof(double)>>>(in_errors, out_errors, n_errors, error_type);
            break;
    }
    CHECK_CUDA(cudaGetLastError());
    // Sum up the blocks in CPU
    CHECK_CUDA(cudaMemcpy(out_errors_host, out_errors, grid_size * sizeof(double), cudaMemcpyDeviceToHost));
    double total = 0;
    for(int k = 0; k < grid_size; k++) {
        total += out_errors_host[k];
    }
    return error_type == RMSE ? sqrt(total / n_errors) : total / n_errors;
}

/** Calculates the total Mean Absolute Error and Root Mean Squared Error on GPU.
 * Needs to call the kernel twice, but is still faster than the CPU version.
 */
std::tuple<float, float> get_error_metrics_gpu(float *in_errors, double *out_errors, double *out_errors_host, int n_errors, int grid_size, int block_size) {
    float mae = calculate_error_metric_gpu(in_errors, out_errors, out_errors_host, n_errors, grid_size, block_size, MAE);
    float rmse = calculate_error_metric_gpu(in_errors, out_errors, out_errors_host, n_errors, grid_size, block_size, RMSE);
    return std::make_tuple(mae, rmse);
}
