#include <stdexcept>
#include <sstream>
#include <iostream>     // std::cout
#include <math.h>       /* pow */

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "matrix.h"
#include "utils.cuh"

#define index(i, j, N)  ((i)*(N)) + (j)

using namespace cu2rec;

// PARALLEL
__global__ void loss_kernel_user(int factors, int user_count, int item_count, const float * P, const float * Q, const int * indptr, const int * indices, const float * data, float * output, float * error) {
    // One thread per user
    int u = blockDim.x * blockIdx.x + threadIdx.x;
    if(u < user_count) {
        // get this user's factors
        const float * p = &P[u * factors];

        for (int i = indptr[u]; i < indptr[u + 1]; ++i) {
            // get this item's factors
            const float * Qi = &Q[indices[i] * factors];

            // update loss with this rating and prediction
            float rating = data[i];

            // TODO: remove hacky dot once we spin our own
            //float pred = cu2rec::dot(Qi, p);

            // TODO: create a better but not hacky dot function, but for now this works well
            float pred = 0.0;
            for (int f = 0; f < factors; f++)
                pred += Qi[f]*p[f];

            // set the error value for this rating
            error[i] = rating - pred;

            // TODO: do we want to do this in kernel here, or will sgd.cu handle the square loss for each error?
            float loss = pow(rating - pred, 2);
            // total_loss += loss;
        }
    }
}

__global__ void total_loss_kernel(float *errors, float *losses, int n_errors, int current_iter) {
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
        losses[current_iter] = errors[0];
    }
}

void calculate_loss_gpu(CudaDenseMatrix* P_d, CudaDenseMatrix* Q_d, int factors, int user_count, int item_count, int num_ratings, CudaCSRMatrix* matrix, float * error_d) {
    // hacky way to store the total loss form kernel so we can retrieve it later
    float loss[1] = {0};
    CudaDenseMatrix* output = new CudaDenseMatrix(1, 1, loss);

    // Dimensions for kernel call
    int n_threads = 32;
    dim3 dimBlock(n_threads);
    dim3 dimGrid(user_count / n_threads + 1);
    loss_kernel_user<<<dimGrid, dimBlock>>>(
        factors, user_count, item_count, P_d->data, Q_d->data,
        matrix->indptr, matrix->indices, matrix->data, output->data, error_d);
    cudaError_t lastError = cudaGetLastError();
    if(cudaSuccess != lastError) {
        printf("ERROR: %s\n", cudaGetErrorName(lastError));
    }
    cudaDeviceSynchronize();

    // move loss output back to host to return
    output->to_host(loss);
}

void calculate_loss_gpu(int factors, int user_count, int item_count, int num_ratings, const float * P, const float * Q, CudaCSRMatrix* matrix, float * error_d) {
    // Turn P and Q into CudaDenseMatrices on GPU
    CudaDenseMatrix* P_d = new CudaDenseMatrix(user_count, factors, P);
    CudaDenseMatrix* Q_d = new CudaDenseMatrix(item_count, factors, Q);
    calculate_loss_gpu(P_d, Q_d, factors, user_count, item_count, num_ratings, matrix, error_d);
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
