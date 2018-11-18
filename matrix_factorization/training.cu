#include <cuda.h>
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
           float **P_ptr, float **Q_ptr, float **losses_ptr) {
    // Initialize P and Q
    float *P = initialize_normal_array(matrix->rows * n_factors);
    float *Q = initialize_normal_array(matrix->cols * n_factors);
    float *losses = new float[n_iterations];
    *P_ptr = P;
    *Q_ptr = Q;
    *losses_ptr = losses;

    // Copy P and Q to device memory
    CudaDenseMatrix* P_device = new CudaDenseMatrix(matrix->rows, n_factors, P);
    CudaDenseMatrix* Q_device = new CudaDenseMatrix(matrix->cols, n_factors, Q);
    CudaDenseMatrix* P_device_target = new CudaDenseMatrix(matrix->rows, n_factors, P);
    CudaDenseMatrix* Q_device_target = new CudaDenseMatrix(matrix->cols, n_factors, Q);

    // Create the errors
    float *errors_device;
    cudaMalloc(&errors_device, matrix->nonzeros * sizeof(float));
    // TODO: see if we can remove this
    cudaMemset(errors_device, 0, matrix->nonzeros * sizeof(float));

    // Create the total losses
    float *losses_device;
    cudaMalloc(&losses_device, n_iterations * sizeof(float));

    // Dimensions
    int n_threads = 32;
    dim3 dim_block(n_threads);
    dim3 dim_grid_sgd(matrix->rows / n_threads + 1);
    dim3 dim_grid_loss(matrix->nonzeros / n_threads + 1);

    // Training loop
    for (int i = 0; i < n_iterations; ++i) {
        calculate_loss_gpu(P_device, Q_device, n_factors, matrix->rows, matrix->cols, matrix->nonzeros,
                           matrix, errors_device);
        sgd_update<<<dim_grid_sgd, dim_block>>>(matrix->indptr, matrix->indices, P_device->data, Q_device->data,
                                                P_device_target->data, Q_device_target->data, n_factors, errors_device,
                                                matrix->rows, matrix->cols, learning_rate);
        total_loss_kernel<<<dim_grid_loss, dim_block>>>(errors_device, losses_device, matrix->nonzeros, i);
        swap(P_device, P_device_target);
        swap(Q_device, Q_device_target);
    }
    
    // Copy array of losses back to host
    cudaMemcpy(losses, losses_device, n_iterations * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy updated P and Q back
    P_device->to_host(P);
    Q_device->to_host(Q);

    cudaFree(errors_device);
    cudaFree(losses_device);
    delete P_device;
    delete P_device_target;
    delete Q_device;
    delete Q_device_target;
}
