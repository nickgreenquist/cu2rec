#include <random>

#include "loss.h"
#include "matrix.h"
#include "sgd.h"

using namespace cu2rec;
using namespace std;

float* initialize_normal_array(int size, float mean, float stddev, int seed) {
    mt19937 generator(seed);
    normal_distribution<float> distribution(mean, stddev);
    float *array = new float[size];
    for(int i = 0; i < size; ++i) {
        array[i] = distribution(generator);
    }
    return array;
}

float* initialize_normal_array(int size, float mean, float stddev) {
    return initialize_normal_array(size, mean, stddev, 42);
}

float* initialize_normal_array(int size, int seed) {
    return initialize_normal_array(size, 0, 1, seed);
}

float *initialize_normal_array(int size) {
    return initialize_normal_array(size, 0, 1);
}

void train(CudaCSRMatrix* matrix, int n_iterations, int n_factors, float learning_rate, int seed,
           float **P_ptr, float **Q_ptr, float **losses_ptr, float **user_bias_ptr, float **item_bias_ptr) {
    int user_count = matrix->rows;
    int item_count = matrix->cols;

    // Initialize P and Q
    float *P = initialize_normal_array(user_count * n_factors);
    float *Q = initialize_normal_array(item_count * n_factors);
    float *losses = new float[n_iterations];
    *P_ptr = P;
    *Q_ptr = Q;
    *losses_ptr = losses;

    // Copy P and Q to device memory
    CudaDenseMatrix* P_device = new CudaDenseMatrix(user_count, n_factors, P);
    CudaDenseMatrix* Q_device = new CudaDenseMatrix(item_count, n_factors, Q);
    CudaDenseMatrix* P_device_target = new CudaDenseMatrix(user_count, n_factors, P);
    CudaDenseMatrix* Q_device_target = new CudaDenseMatrix(item_count, n_factors, Q);

    // Create the errors
    float *errors_device;
    cudaMalloc(&errors_device, matrix->nonzeros * sizeof(float));

    // Create the total losses
    float *losses_device;
    cudaMalloc(&losses_device, n_iterations * sizeof(float));

    // Create the bias arrays
    float *user_bias = initialize_normal_array(user_count);
    float *item_bias = initialize_normal_array(item_count);
    *user_bias_ptr = user_bias;
    *item_bias_ptr = item_bias;
    
    float *user_bias_device;
    cudaMalloc(&user_bias_device, user_count * sizeof(float));
    cudaMemcpy(user_bias_device, user_bias, user_count * sizeof(float), cudaMemcpyHostToDevice);

    float *item_bias_device;
    cudaMalloc(&item_bias_device, item_count * sizeof(float));
    cudaMemcpy(item_bias_device, item_bias, item_count * sizeof(float), cudaMemcpyHostToDevice);

    // Create bias targets
    float *user_bias_target, *item_bias_target;
    cudaMalloc(&user_bias_target, user_count * sizeof(float));
    cudaMemcpy(user_bias_target, user_bias, user_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&item_bias_target, item_count * sizeof(float));
    cudaMemcpy(item_bias_target, item_bias, item_count * sizeof(float), cudaMemcpyHostToDevice);

    // Dimensions
    int n_threads = 32;
    dim3 dim_block(n_threads);
    dim3 dim_grid_sgd(user_count / n_threads + 1);
    dim3 dim_grid_loss(matrix->nonzeros / n_threads + 1);

    // Training loop
    cudaError_t lastError;
    for (int i = 0; i < n_iterations; ++i) {
        // Calculate initial error per each rating
        calculate_loss_gpu(P_device, Q_device, n_factors, user_count, item_count, matrix->nonzeros, matrix,
                           errors_device, user_bias_device, item_bias_device);

        // Run single iteration of SGD
        sgd_update<<<dim_grid_sgd, dim_block>>>(matrix->indptr, matrix->indices, P_device->data, Q_device->data,
                                                P_device_target->data, Q_device_target->data, n_factors, errors_device,
                                                user_count, item_count, learning_rate, user_bias_device, item_bias_device,
                                                user_bias_target, item_bias_target, learning_rate, learning_rate);
        lastError = cudaGetLastError();
        if(cudaSuccess != lastError) {
            printf("ERROR: %s\n", cudaGetErrorName(lastError));
        }

        // Calculate total loss to check for improving loss
        total_loss_kernel<<<dim_grid_loss, dim_block>>>(errors_device, losses_device, matrix->nonzeros, i);
        lastError = cudaGetLastError();
        if(cudaSuccess != lastError) {
            printf("ERROR: %s\n", cudaGetErrorName(lastError));
        }

        // Swap old and new P and Q
        swap(P_device, P_device_target);
        swap(Q_device, Q_device_target);

        // Swap old and new bias arrays
        swap(user_bias_device, user_bias_target);
        swap(item_bias_device, item_bias_target);
    }
    
    // Copy array of losses back to host
    cudaMemcpy(losses, losses_device, n_iterations * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy updated P and Q back
    P_device->to_host(P);
    Q_device->to_host(Q);

    // Copy updated bias arrays back
    cudaMemcpy(user_bias, user_bias_device, user_count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(item_bias, item_bias_device, item_count * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(errors_device);
    cudaFree(losses_device);
    cudaFree(user_bias_device);
    cudaFree(item_bias_device);
    cudaFree(user_bias_target);
    cudaFree(item_bias_target);
    delete P_device;
    delete P_device_target;
    delete Q_device;
    delete Q_device_target;
}
