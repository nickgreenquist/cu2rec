#include <stdexcept>
#include <sstream>
#include <iostream>     // std::cout
#include <math.h>       /* pow */

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "matrix.h"

#define index(i, j, N)  ((i)*(N)) + (j)

using namespace cu2rec;

// PARALLEL
__global__ void loss_kernel(int factors, int user_count, int item_count, const float * P, const float * Q, const int * indptr, 
                            const int * indices, const float * data, float * error, float * user_bias, float * item_bias, float global_bias) {
    // One thread per user
    int u = blockDim.x * blockIdx.x + threadIdx.x;
    if(u < user_count) {
        // get this user's factors
        const float * p = &P[u * factors];

        for (int i = indptr[u]; i < indptr[u + 1]; ++i) {
            // get this item's factors
            int item_id = indices[i];
            const float * Qi = &Q[item_id * factors];

            // calculate predicted rating
            float pred = global_bias + user_bias[u] + item_bias[item_id];
            for (int f = 0; f < factors; f++)
                pred += Qi[f]*p[f];

            // set the error value for this rating: rating - pred
            error[i] = data[i] - pred;
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
    loss_kernel<<<dimGrid, dimBlock>>>(
        factors, user_count, item_count, P_d->data, Q_d->data,
        matrix->indptr, matrix->indices, matrix->data, error_d,
        user_bias, item_bias, global_bias);
    cudaError_t lastError = cudaGetLastError();
    if(cudaSuccess != lastError) {
        printf("ERROR: %s\n", cudaGetErrorName(lastError));
    }
}

// SEQUENTIAL
float dot_product_sequential(const float *Qi, const float *p, int n) {
    float result = 0.0;
    for (int i = 0; i < n; i++)
        result += Qi[i]*p[i];
    return result;
}
float calculate_loss_sequential(int factors, int user_count, int item_count, const float * P, const float * Q, const int * indptr, const int * indices, const float * data) {
    float total_loss = 0;
    for(int u = 0; u < user_count; u++) {
        // get this user's factors
        float *p = new float[factors];
        for(int f = 0; f < factors; f++) {
            p[f] = P[index(u, f, factors)];
        }

        for (int i = indptr[u]; i < indptr[u + 1]; ++i) {
            // get this item's factors
            float *Qi = new float[factors];
            int item_id = indices[i];
            for(int f = 0; f < factors; f++) {
                Qi[f] = Q[index(item_id, f, factors)];
            }

            // update loss with this rating and prediction
            float rating = data[i];
            float pred = dot_product_sequential(Qi, p, factors);

            // std::cout << "Rating: " << rating << ", Pred: " << pred << "\n";

            float loss = pow(rating - pred, 2);
            total_loss += loss;

            delete [] Qi;
        }
        delete [] p;
    }
    return total_loss;
}
