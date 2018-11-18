#include <stdexcept>
#include <sstream>
#include <iostream>     // std::cout
#include <math.h>       /* pow */

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "matrix.h"
#include "utils.cuh"

#define index(i, j, N)  ((i)*(N)) + (j)

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

void calculate_loss_gpu(int factors, int user_count, int item_count, int num_ratings, const float * P, const float * Q, cu2rec::CudaCSRMatrix* matrix, float * error) {
    // Turn P and Q into CudaDenseMatrices on GPU
    cu2rec::CudaDenseMatrix* P_d = new cu2rec::CudaDenseMatrix(user_count, factors, P);
    cu2rec::CudaDenseMatrix* Q_d = new cu2rec::CudaDenseMatrix(item_count, factors, Q);

    // hacky way to store the total loss form kernel so we can retrieve it later
    float loss[1] = {0};
    cu2rec::CudaDenseMatrix* output = new cu2rec::CudaDenseMatrix(1, 1, loss);

    // make copy of error array
    float* error_d;
    cudaMalloc((void **) &error_d, num_ratings * sizeof(float));
    cudaMemset(error_d, 0, num_ratings * sizeof(float));

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

    // move array of errors back to host
    cudaMemcpy(error, error_d, num_ratings * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(error_d);
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