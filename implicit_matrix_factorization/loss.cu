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
__global__ void loss_kernel(int factors, int user_count, int item_count, const float * P, const float * Q, const int * indptr, const int * indices, const float * data, float * output, float * error) {
    float total_loss = 0.0;
    for (int u = blockIdx.x; u < user_count; u += gridDim.x) {
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
            float pred = cu2rec::dot(Qi, p);

            error[i] = rating - pred;
            float loss = pow(rating - pred, 2);
            total_loss += loss;

            delete [] Qi;
        }
        delete [] p;
    }

    if (threadIdx.x == 0) {
        atomicAdd(output, total_loss);
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

    // call the kernel with hardcoded dimensions for now
    // TODO: input better dimensions
    loss_kernel<<<1024, factors, sizeof(float) * factors>>>(
        factors, user_count, item_count, P_d->data, Q_d->data,
        matrix->indptr, matrix->indices, matrix->data, output->data, error_d);
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